# from lilab.timecode_tag.decorder import getDecoder
import numpy as np
from torch2trt import TRTModule
from torch2trt.torch2trt import torch_dtype_from_trt
import torch
import os.path as osp

class Decoder:
    def __init__(self, model_path:str, igpu:int = 0):
        with torch.cuda.device(f'cuda:{igpu}'):
            trt_model = TRTModule()
            trt_model.load_from_engine(model_path)

            IN_list = []
            self.argsort = np.argsort(trt_model.output_names)

            for i in range(len(trt_model.input_names)):
                name = trt_model.input_names[i]
                input_dtype = torch_dtype_from_trt(trt_model.engine.get_tensor_dtype(name))
                input_shape = np.array(trt_model.context.get_tensor_shape(name))
                input_shape[input_shape<=0] = 1
                img_NCHW = np.zeros(input_shape)
                batch_img = torch.from_numpy(img_NCHW).cuda().type(input_dtype)
                IN_list.append(batch_img)

            # warm up
            result = trt_model(*IN_list)
            self.input_shape = tuple(IN_list[0].shape)
            self.trt_model = trt_model
            
    def __call__(self, img:np.ndarray):
        img = img.squeeze()
        assert img.shape[0] >= self.input_shape[0] and img.shape[1] >= self.input_shape[1]
        img = np.ascontiguousarray(img[:self.input_shape[0], :self.input_shape[1]])
        img_torch = torch.from_numpy(img).cuda().float()
        result = self.trt_model(img_torch)
        result_sort = [result[i] for i in self.argsort]
        timecode = result_sort[0].item()
        pvalue = result_sort[1].cpu().numpy()
        return timecode, pvalue

__decoder = None

def getDecoder() -> Decoder:
    global __decoder
    if __decoder is None:
        engine = osp.join(osp.dirname(__file__), 'shufflenet_120_norm.engine')
        __decoder = Decoder(engine)
    return __decoder


if __name__ == '__main__':
    import ffmpegcv
    import tqdm
    vid = ffmpegcv.VideoCapture('time_code.mp4', pix_fmt='gray')
    decoder = Decoder('resnet18_norm.engine')
    timecode_l = np.zeros((len(vid),), dtype=int)
    pvalue_l = np.zeros((len(vid),3), dtype=float)
    for i, frame in enumerate(tqdm.tqdm(vid)):
        timecode, pvalue = decoder(frame.squeeze())
        timecode_l[i] = timecode
        pvalue_l[i] = pvalue

