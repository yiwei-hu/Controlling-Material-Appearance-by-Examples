import random
import torch


def mycrop(x, size, center=False, rand0=None, tileable=True):

    b,c,h,w = x.shape
    if center:
        w0 = (w - size)*0.5
        h0 = (h - size)*0.5
        w0 = int(w0)
        h0 = int(h0)

        if w0+size>w or h0+size>h:
            raise ValueError('value error of w0')

        return x[:,:,w0:w0+size,h0:h0+size]

    if not tileable:
        if rand0 is None:
            w0 = random.randint(0, w-size)
            h0 = random.randint(0, h-size)
        else:
            h0 = rand0[0]
            w0 = rand0[1]

        if w0+size>w or h0+size>h:
            raise ValueError('value error of w0')

        return x[:,:,w0:w0+size,h0:h0+size]

    else:
        if rand0 is None:
            w0 = random.randint(-size+1,w-1)
            h0 = random.randint(-size+1,h-1)
        else:
            h0 = rand0[0]
            w0 = rand0[1]

        wc = w0 + size
        hc = h0 + size

        p = torch.ones((b,c,size,size), device='cuda')

        # seperate crop and stitch them manually
        # [7 | 8 | 9]
        # [4 | 5 | 6]
        # [1 | 2 | 3]
        # 1
        if h0<=0 and w0<=0:
            p[:,:,0:-h0,0:-w0] = x[:,:, h+h0:h, w+w0:w]
            p[:,:,-h0:,0:-w0] = x[:,:, 0:hc, w+w0:w]
            p[:,:,0:-h0,-w0:] = x[:,:, h+h0:h, 0:wc]
            p[:,:,-h0:,-w0:] = x[:,:, 0:hc, 0:wc]
        # 2
        elif h0<=0 and (w0<w-size and w0>0):
            p[:,:,0:-h0,:] = x[:,:, h+h0:h,w0:wc]
            p[:,:,-h0:,:] = x[:,:, 0:hc, w0:wc]
        # 3
        elif h0<=0 and w0 >=w-size:
            p[:,:,0:-h0,0:w-w0] = x[:,:, h+h0:h, w0:w]
            p[:,:,-h0:,0:w-w0] = x[:,:, 0:hc, w0:w]
            p[:,:,0:-h0,w-w0:] = x[:,:, h+h0:h, 0:wc-w]
            p[:,:,-h0:,w-w0:] = x[:,:, 0:hc, 0:wc-w]

        # 4
        elif (h0>0 and h0<h-size) and w0<=0:
            p[:,:,:,0:-w0] = x[:,:, h0:hc, w+w0:w]
            p[:,:,:,-w0:] = x[:,:, h0:hc, 0:wc]
        # 5
        elif (h0>0 and h0<h-size) and (w0<w-size and w0>0):
            p = x[:,:, h0:hc, w0:wc]
        # 6
        elif (h0>0 and h0<h-size) and w0 >=w-size:
            p[:,:,:,0:w-w0] = x[:,:, h0:hc, w0:w]
            p[:,:,:,w-w0:] = x[:,:, h0:hc, 0:wc-w]

        # 7
        elif h0 >=h-size and w0<=0:
            p[:,:,0:h-h0,0:-w0] = x[:,:, h0:h, w+w0:w]
            p[:,:,h-h0:,0:-w0] = x[:,:, 0:hc-h, w+w0:w]
            p[:,:,0:h-h0,-w0:] = x[:,:, h0:h, 0:wc]
            p[:,:,h-h0:,-w0:] = x[:,:, 0:hc-h, 0:wc]
        # 8
        elif h0 >=h-size and (w0<w-size and w0>0):
            p[:,:,0:h-h0,:] = x[:,:, h0:h,w0:wc]
            p[:,:,h-h0:,:] = x[:,:, 0:hc-h, w0:wc]
        # 9
        elif h0 >=h-size and w0 >=w-size:
            p[:,:,0:h-h0,0:w-w0] = x[:,:, h0:h, w0:w]
            p[:,:,h-h0:,0:w-w0] = x[:,:, 0:hc-h, w0:w]
            p[:,:,0:h-h0,w-w0:] = x[:,:, h0:h, 0:wc-w]
            p[:,:,h-h0:,w-w0:] = x[:,:, 0:hc-h, 0:wc-w]

        del x

        return p
