import numpy as np
import os
from argparse import ArgumentParser
from tensorflow.keras.models import load_model
import utils as U

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--sub", type=int, required=True)
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--gpu", type=int, required=False, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    U.set_gpu(args.gpu)

    data_test = np.load("/nas/home/fpicetti/datasets/seismic_facies/2020_aicrowd_challenge/data_test_1.npz",
                        allow_pickle=True, mmap_mode='r')
    data_test = data_test['data']
    data_min, data_max = data_test.min(), data_test.max()
    data_shape = data_test.shape
    xz_shape = data_shape[:2]
    yz_shape = (data_shape[0], data_shape[2])
    
    run = np.load(os.path.join('./results/', args.run, "run.npy"), allow_pickle=True).item()
    
    # load model
    model = load_model(os.path.join('./results/', args.run, '_model.h5'))
    model.load_weights(os.path.join('./results/', args.run, '_weights.h5'))
    input_shape = model.input_shape[1:-1]
    
    # predict xz
    pe_xz = U.PatchExtractor(input_shape, stride=())
    xz_shape_out = U.in_content_cropped_shape(xz_shape, pe_xz.dim[:2], pe_xz.stride[:2])

    out_xz_patches = []
    for inline in range(data_test.shape[2]):
        patches = pe_xz.extract(data_test[:, :, inline]).reshape((-1,) + pe_xz.dim + (1,))
        out_xz_patches.append(model.predict(patches))

    # predict yz
    pe_yz = U.PatchExtractor(input_shape)
    yz_shape_out = U.in_content_cropped_shape(yz_shape, pe_yz.dim[:2], pe_yz.stride[:2])

    out_yz_patches = []
    for inline in range(data_test.shape[2]):
        patches = pe_yz.extract(data_test[:, :, inline]).reshape((-1,) + pe_yz.dim + (1,))
        out_yz_patches.append(model.predict(patches))

    # post processing
    
    
    outpath = os.path.join('./results/', 'facies_aicrowd', 'submission%s.npz' % str(args.sub).zfill(3))
    np.savez_compressed(outpath, prediction=prediction)
    
    print("Test done! Saved to %s" % outpath)
