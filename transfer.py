import random
from torch.optim import Adam, lr_scheduler
from loss import *
from utils import *
from renderer import *
from TileableMaterialGAN import TileableMaterialGAN


# project SV-BRDF maps into the latent space of MaterialGAN
def project(args, generator, svbrdf_ref):
    out_dir = pth.join(args.in_dir, 'latent')
    out_tmp_dir = os.path.join(out_dir, 'tmp')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_tmp_dir, exist_ok=True)

    # optimization params
    n_step = args.proj_steps
    lr_start_decay = 1000
    lr = args.proj_lr
    tolerance = 0.02
    equal_weight_at = 1000

    # initialize renderer
    renderer = Renderer(args.im_res, preload='point', optimizable=False)
    with torch.no_grad():
        render_and_save(svbrdf_ref, out_dir, out_tmp_dir, 0, [renderer], suffix='_gt')

    lossObj = EmbeddingLosses(args, svbrdf_ref, renderer=renderer, scales=(0, 1))

    optimizable = generator.get_optimizable()
    optimizer = Adam(optimizable, lr=lr, betas=(0.9, 0.999))
    lr_lambda = lambda x: 1 if x <= lr_start_decay else (n_step - x) / (n_step - lr_start_decay)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)

    min_loss = np.inf
    loss_list = {}

    print('*'*10 + 'Projecting to latent space ' + '*'*10)
    for step in range(n_step):
        optimizer.zero_grad()

        # evaluate material maps
        svbrdf = generator()

        # compute loss
        alpha = max(0.0, (equal_weight_at - step) / equal_weight_at)
        loss, losses = lossObj.eval(svbrdf, alpha=alpha)
        record_loss(loss_list, losses)

        # update latent
        loss.backward()
        optimizer.step()
        scheduler.step()

        # save output
        if step % 100 == 0 or step == n_step - 1:
            print(f'[{step + 1}/{n_step}], pixel loss = {losses["pixel"]:.4f}, feature loss = {losses["feature"]:.4f}, '
                  f'render loss = {losses["render"]:.4f}, loss = {losses["tot"]:.4f}, lr = {scheduler.get_last_lr()}')

            with torch.no_grad():
                render_and_save(svbrdf, out_dir, out_tmp_dir, step, [renderer])

            plot_and_save(loss_list, pth.join(out_tmp_dir, 'loss.png'))

            if loss.item() < min_loss:
                print('Saving optimal latent vector and noises.')
                min_loss = loss.item()
                generator.save(out_dir, prefix='init_')

        # if loss is small enough, then stop early
        if loss.item() < tolerance:
            break


# transfer appearance from target images to SV-BRDF maps
def transfer(args):
    np.set_printoptions(precision=4, suppress=True)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    out_dir = args.out_dir
    out_tmp_dir = os.path.join(out_dir, 'tmp')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_tmp_dir, exist_ok=True)

    save_args(args, os.path.join(out_dir, 'args.txt'))
    for k, v in args.__dict__.items():
        print(f'{k}: {v}')

    # load data
    input_svbrdf, target_images, labels = MaterialLoader(args.im_res, args.device).load(args.in_dir, invert_normal=args.invert_normal)
    input_svbrdf = MaterialLoader.remapping(input_svbrdf)
    guidance_maps = compute_guidance_maps(labels, out_dir)

    # save data
    save_material_maps(input_svbrdf, out_dir, suffix='_input')
    if target_images.shape[0] == 1:
        save_image(target_images, pth.join(out_dir, 'target.png'))
    else:
        for i in range(target_images.shape[0]):
            save_image(target_images[i: i+1], pth.join(out_dir, f'target{i}.png'))

    # initialize renderer
    renderer = Renderer(args.im_res, light_config=args.light_and_camera, optimizable=False)
    renderer.print_light_config()
    render_and_save(input_svbrdf, out_dir, out_tmp_dir, 0, [renderer], suffix='_input')

    # initialize generator
    generator = TileableMaterialGAN(args)
    if generator.latent_type == 'random_mean' and not args.rand_init:
        project(args, generator, input_svbrdf)

    # define loss functions
    lossObj = StyleTransferLosses(args, input_svbrdf, target_images, None, renderer, guidance_maps=guidance_maps, scales=(0, 1, 2))

    # define optimizer
    optimizable = generator.get_optimizable()
    optimizer = Adam(optimizable + list(renderer.parameters()), lr=args.lr, betas=(0.9, 0.999))

    # output initial material maps
    with torch.no_grad():
        init_svbrdf = generator().detach()
        save_material_maps(init_svbrdf, out_dir, suffix='_init')
        render_and_save(init_svbrdf, out_dir, out_tmp_dir, 0, [renderer], suffix='_init')

    min_loss = np.inf
    loss_list = {}

    for step in range(args.steps):
        optimizer.zero_grad()

        # evaluate material maps
        svbrdf = generator()
        loss, losses = lossObj.eval(svbrdf)
        record_loss(loss_list, losses)

        # update latent
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            renderer.regularize()

        # save output
        if step % args.vis_every == 0 or step == args.steps - 1:
            print(f'[{step + 1}/{args.steps}], style loss = {losses["style"]}, '
                  f'content loss = {losses["content"]}, loss = {loss.item()}')
            if renderer.optimizable:
                print(f"Current light position: {renderer.lp.data}")
                print(f"Current camera position: {renderer.cp.data}")
                print(f"Current light intensity: {renderer.li.data}")

            with torch.no_grad():
                svbrdf_cur = generator().detach()
                render_and_save(svbrdf_cur, out_dir, out_tmp_dir, step, [renderer])

            plot_and_save(loss_list, pth.join(out_tmp_dir, 'loss.png'))

            if loss.item() < min_loss:
                min_loss = loss.item()
                save_material_maps(svbrdf_cur, out_dir, suffix='_optim')
                generator.save(out_dir, prefix='optim_')
