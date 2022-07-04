import math
import random
import torch
from torch import nn
from torch.nn import functional as F
from models.building_blocks import PixelNorm, EqualLinear, ConstantInput, StyledConv, ToRGB


class Generator(nn.Module):
    def __init__(self, args, device, blur_kernel=[1, 3, 3, 1], lr_mlp=0.01):
        super().__init__()

        self.args = args
        self.device = device

        layers = [PixelNorm()]
        for i in range(args.n_mlp):
            layers.append(EqualLinear(args.style_dim, args.style_dim, lr_mul=lr_mlp, activation='fused_lrelu' )  )
        self.style = nn.Sequential(*layers)

        if self.args.condv=='3' or self.args.condv =='4':
            print('having condition mapping')
            layers = [PixelNorm()]
            for i in range(args.n_mlp):
                if i==0:
                    layers.append( EqualLinear( 10, args.style_dim, lr_mul=lr_mlp, activation='fused_lrelu' )  )
                else:
                    layers.append( EqualLinear( args.style_dim, args.style_dim, lr_mul=lr_mlp, activation='fused_lrelu' )  )
            self.style_c = nn.Sequential(*layers)            

        self.channels = {   4: 512,
                            8: 512,
                            16: 512,
                            32: 512,
                            64: int(256 * args.channel_multiplier),
                            128: int(128 * args.channel_multiplier),
                            256: int(64 * args.channel_multiplier),
                            512: int(32 * args.channel_multiplier),
                            1024: int(16 * args.channel_multiplier) }

        if self.args.condv=='4':
            self.inject_index=10

        final_channel = args.nc

        if self.args.no_cond or self.args.condv!='1':
            self.input = ConstantInput(self.channels[4])
        else:
            raise NotImplementedError

        self.w_over_h = args.scene_size[1] / args.scene_size[0]
        assert self.w_over_h.is_integer(), 'non supported scene_size'
        self.w_over_h = int(self.w_over_h)

        self.log_size = int(math.log(args.scene_size[0], 2)) - int(math.log(self.args.starting_height_size, 2))
        self.num_layers = self.log_size * 2 
        self.n_latent = self.log_size * 2 + 1 

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        expected_out_size = self.args.starting_height_size
        layer_idx = 0 
        for _ in range(self.log_size):
            expected_out_size *= 2
            shape = [1, 1, expected_out_size, expected_out_size*self.w_over_h]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.zeros(*shape))
            self.noises.register_buffer(f'noise_{layer_idx+1}', torch.zeros(*shape))
            layer_idx += 2 

        in_channel = self.channels[self.args.starting_height_size]
        expected_out_size = self.args.starting_height_size     
        in_style_dim = args.style_dim*2 if self.args.condv=='3' else args.style_dim
        for _ in range(self.log_size):  
            expected_out_size *= 2 
            out_channel = self.channels[expected_out_size]
            self.convs.append(StyledConv( in_channel, out_channel, 3, in_style_dim, upsample=True, blur_kernel=blur_kernel, circular=args.circular, circular2=args.circular2))
            self.convs.append(StyledConv(out_channel, out_channel, 3, in_style_dim, blur_kernel=blur_kernel, circular=args.circular, circular2=args.circular2))
            self.to_rgbs.append(ToRGB(out_channel, in_style_dim, out_channel=final_channel, circular=args.circular, circular2=args.circular2))
            in_channel = out_channel                               

    def make_noise(self):
        expected_out_size = self.args.starting_height_size
        noises = []
        for _ in range(self.log_size):
            expected_out_size *= 2
            noises.append( torch.randn(1, 1, expected_out_size, expected_out_size*self.w_over_h, device=self.device) )
            noises.append( torch.randn(1, 1, expected_out_size, expected_out_size*self.w_over_h, device=self.device) )

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn( n_latent, self.args.style_dim, device=self.device)
        latent = self.style(latent_in).mean(0, keepdim=True).unsqueeze(1)
        return latent

    def get_latent(self, input):
        return self.style(input)

    def __prepare_starting_feature(self, global_pri, styles, input_type, jitter):
        if self.args.no_cond:
            # print(global_pri.shape[0])
            feature = self.input(global_pri)
            z = self.args.truncate_z * torch.randn(global_pri.shape[0], self.args.style_dim, device=self.device)
            loss = torch.tensor([0.0], requires_grad=True, device=self.device)
        else:
            if self.args.condv=='2':
                feature = self.input(global_pri)
                input_std, input_m = torch.std_mean(global_pri, dim=(2,3))
                input_cat = torch.cat([input_m, input_std], dim=1)
                z = torch.randn(global_pri.shape[0], self.args.style_dim-10, device=self.device)
                z = torch.cat([z, input_cat], dim=1)
                loss = torch.tensor([0.0], requires_grad=True, device=self.device)
            # c --> map_c --> w_c cat w
            elif self.args.condv=='3' or self.args.condv=='4':
                feature = self.input(global_pri)
                z = self.args.truncate_z * torch.randn(global_pri.shape[0], self.args.style_dim, device=self.device)
                loss = torch.tensor([0.0], requires_grad=True, device=self.device)
            else:
                feature, z, loss = self.encoder(global_pri, jitter=jitter)
        if input_type is None:
            styles = [z]
            input_type = 'z'
        return feature, styles, input_type, loss

    def __prepare_letent(self, styles, inject_index, truncation, truncation_latent,  input_type, style_c=None):
        "This is a private function to prepare w+ space code needed during forward"
        if input_type == 'z':
            styles = [self.style(s).unsqueeze(1) for s in styles]  # each one is bs*1*512
            if self.args.condv=='3':
                input_std, input_m = torch.std_mean(style_c, dim=(2,3))
                input_cat = torch.cat([input_m, input_std], dim=1)
                style_c = [self.style_c(input_cat).unsqueeze(1)]  # each one is bs*1*512
                styles = [torch.cat([s,c], dim=-1) for s,c in zip(styles,style_c)]

            elif self.args.condv=='4':
                input_std, input_m = torch.std_mean(style_c, dim=(2,3))
                input_cat = torch.cat([input_m, input_std], dim=1)
                style_c = [self.style_c(input_cat).unsqueeze(1)]  # each one is bs*1*512

        elif input_type == 'w':
            styles = [s.unsqueeze(1) for s in styles]  # each one is bs*1*512
        else: 
            return styles

        # truncate each w 
        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append( truncation_latent + truncation * (style-truncation_latent)  )
            styles = style_t

        # duplicate and concat into BS * n_latent * code_len 
        if len(styles) == 1:
            if self.args.condv=='4':
                latent = styles[0].repeat(1, self.n_latent - self.inject_index, 1) 
                latent_c = style_c[0].repeat(1, self.inject_index, 1) 
                print('mixing cond style')
                latent = torch.cat([latent, latent_c], 1)
            else:
                latent = styles[0].repeat(1, self.n_latent, 1) 

        elif len(styles) == 2:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)
            latent1 = styles[0].repeat(1, inject_index, 1)
            latent2 = styles[1].repeat(1, self.n_latent - inject_index, 1)
            latent = torch.cat([latent1, latent2], 1)
        else:
            latent = torch.cat(styles, 1)

        return latent

    def __prepare_noise(self, noise, randomize_noise):
        "This is a private function to prepare noise needed during forward"

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [ getattr(self.noises, f'noise_{i}') for i in range(self.num_layers) ]

        return noise
  
    def forward(self, global_pri, styles=None, return_latents=False, inject_index=None, truncation=1, truncation_latent=None, input_type=None, noise=None, randomize_noise=True, return_loss=True, shiftN=None, jitter=None):

        """
        global_pri: a tensor with the shape BS*C*self.prior_size*self.prior_size. Here, in background training,
                    it should be semantic map + edge map, so it should have channel 151+1 

        styles: it should be either list or a tensor or None.
                List Case:
                    a list containing either z code or w code.
                    and each code (whether z or w, specify by input_type) in this list should be bs*len_of_code.
                    And number of codes should be 1 or 2 or self.n_latent. 
                    When len==1, later this code will be broadcast into bs*self.n_latent*512
                    if it is 2 then it will perform style mixing. If it is self.n_latent, then each of them will 
                    provide style for each layer.
                Tensor Case:
                    then it has to be bs*self.n_latent*code_len, which means it is a w+ code.
                    In this case input_type should be 'w+', and for now we do not support truncate,
                    we assume the input is a ready-to-go latent code from w+ space
                None Case:
                    Then z code will be derived from global_pri also. In this case input_type shuold be None
            
        return_latents: if true w+ code: bs*self.n_latent*512 tensor, will be returned 

        inject_index: int value, it will be specify for style mixing, only will be used when len(styles)==2 

        truncation: whether each w will be truncated 
        
        truncation_latent: if given then it should be calculated from mean_latent function. It has size 1*1*512
                           if truncation, then this latent must be given 

        input_type: input type of styles, None, 'z', 'w' 'w+'
        
        noise: if given then recommand to run make_noise first to get noise and then use that as input. if given 
               randomize_noise will be ignored 
         
        randomize_noise: if true then each forward will use different noise, if not a pre-registered fixed noise
                         will be used for each forward.

        return_loss: if return kl loss.  

        """
        if input_type == 'z' or input_type == 'w': 
            assert len(styles) in [1,2,self.n_latent], f'number of styles must be 1, 2 or self.n_latent but got {len(styles)}'
        elif input_type == 'w+':
            assert styles.ndim == 3 and styles.shape[1] == self.n_latent
        elif input_type is None:
            assert styles is None
        else:
            assert False, 'not supported input_type'

        start_feature, styles, input_type, loss = self.__prepare_starting_feature(global_pri, styles, input_type, jitter)
        latent = self.__prepare_letent(styles, inject_index, truncation, truncation_latent, input_type, style_c=global_pri if self.args.condv=='3' or self.args.condv=='4' else None)
        noise = self.__prepare_noise(noise, randomize_noise)

        out = start_feature
        skip = None

        assert out.isnan().any()==False, 'start_feature nan'

        i = 0
        for conv1, conv2, noise1, noise2, to_rgb in zip( self.convs[::2], self.convs[1::2], noise[::2], noise[1::2], self.to_rgbs ):
            out = conv1(out, latent[:, i], noise=noise1, shiftN=shiftN)  
            assert out.isnan().any()==False, 'out1 nan'

            out = conv2(out, latent[:, i + 1], noise=noise2, shiftN=shiftN)
            assert out.isnan().any()==False, 'out2 nan'

            skip = to_rgb(out, latent[:, i + 2], skip)
            assert skip.isnan().any()==False, 'skip nan'

            i += 2

        image = F.tanh(skip)

        output = {'image': image}
        if return_latents:
            output['latent'] =  latent  
        if return_loss:
            output['klloss'] =  loss 

        return output