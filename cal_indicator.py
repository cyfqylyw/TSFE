import numpy as np
import os
import cv2
import time
import skimage
import phasepack.phasecong as pc
import torch
import torch.nn.functional as F


def cal_ssim(img1, img2):
    """
    Calculate the Structural Similarity Index (SSIM) between two images, 
    :img1: the first image, an np.array
    :img2: the second image, an np.array
    """
    # Convert the images to grayscale
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Compute the Structural Similarity Index (SSIM)
    return skimage.metrics.structural_similarity(gray_img1, gray_img2, win_size=11, use_sample_covariance=True)


def _gradient_magnitude(img: np.ndarray, img_depth):
    """
    Calculate gradient magnitude based on Scharr operator
    """
    scharrx = cv2.Scharr(img, img_depth, 1, 0)
    scharry = cv2.Scharr(img, img_depth, 0, 1)

    return np.sqrt(scharrx ** 2 + scharry ** 2)


def _similarity_measure(x, y, constant):
    """
    Calculate feature similarity measurement between two images
    """
    numerator = 2 * x * y + constant
    denominator = x ** 2 + y ** 2 + constant

    return numerator / denominator


def cal_fsim(org_img: np.ndarray, pred_img: np.ndarray, T1=0.85, T2=160) -> float:
    """
    Feature-based similarity index, based on phase congruency (PC) and image gradient magnitude (GM)

    There are different ways to implement PC, the authors of the original FSIM paper use the method
    defined by Kovesi (1999). The Python phasepack project fortunately provides an implementation
    of the approach.

    There are also alternatives to implement GM, the FSIM authors suggest to use the Scharr
    operation which is implemented in OpenCV.

    Note that FSIM is defined in the original papers for grayscale as well as for RGB images. Our use cases
    are mostly multi-band images e.g. RGB + NIR. To accommodate for this fact, we compute FSIM for each individual
    band and then take the average.

    Note also that T1 and T2 are constants depending on the dynamic range of PC/GM values. In theory this parameters
    would benefit from fine-tuning based on the used data, we use the values found in the original paper as defaults.

    Args:
        org_img -- numpy array containing the original image
        pred_img -- predicted image
        T1 -- constant based on the dynamic range of PC values
        T2 -- constant based on the dynamic range of GM values
    """

    alpha = beta = 1  # parameters used to adjust the relative importance of PC and GM features
    fsim_list = []
    for i in range(org_img.shape[2]):
        # Calculate the PC for original and predicted images
        pc1_2dim = pc(org_img[:, :, i], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978)
        pc2_2dim = pc(pred_img[:, :, i], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978)

        # pc1_2dim and pc2_2dim are tuples with the length 7, we only need the 4th element which is the PC.
        # The PC itself is a list with the size of 6 (number of orientation). Therefore, we need to
        # calculate the sum of all these 6 arrays.
        pc1_2dim_sum = np.zeros((org_img.shape[0], org_img.shape[1]), dtype=np.float64)
        pc2_2dim_sum = np.zeros((pred_img.shape[0], pred_img.shape[1]), dtype=np.float64)
        for orientation in range(6):
            pc1_2dim_sum += pc1_2dim[4][orientation]
            pc2_2dim_sum += pc2_2dim[4][orientation]

        # Calculate GM for original and predicted images based on Scharr operator
        gm1 = _gradient_magnitude(org_img[:, :, i], cv2.CV_16U)
        gm2 = _gradient_magnitude(pred_img[:, :, i], cv2.CV_16U)

        # Calculate similarity measure for PC1 and PC2
        S_pc = _similarity_measure(pc1_2dim_sum, pc2_2dim_sum, T1)
        # Calculate similarity measure for GM1 and GM2
        S_g = _similarity_measure(gm1, gm2, T2)

        S_l = (S_pc ** alpha) * (S_g ** beta)

        numerator = np.sum(S_l * np.maximum(pc1_2dim_sum, pc2_2dim_sum))
        denominator = np.sum(np.maximum(pc1_2dim_sum, pc2_2dim_sum))
        fsim_list.append(numerator / denominator)

    return np.mean(fsim_list)


def cal_gmsd(dis_img, ref_img, c=170, device='mps'):
    # check for input type
    if type(dis_img) == np.ndarray:
        assert dis_img.ndim == 2 or dis_img.ndim == 3
        if dis_img.ndim == 2:
            dis_img = torch.from_numpy(dis_img).unsqueeze(0).unsqueeze(0)
        else:
            dis_img = torch.from_numpy(dis_img).unsqueeze(0)

    if type(ref_img) == np.ndarray:
        assert ref_img.ndim == 2 or ref_img.ndim == 3
        if ref_img.ndim == 2:
            ref_img = torch.from_numpy(ref_img).unsqueeze(0).unsqueeze(0)
        else:
            ref_img = torch.from_numpy(ref_img).unsqueeze(0)
    # input is a gray-image, with pixel between 0 - 255
    if torch.max(dis_img) <= 1:
        dis_img = dis_img * 255
    if torch.max(ref_img) <= 1:
        ref_img = ref_img * 255

    hx=torch.tensor([[1/3,0,-1/3]]*3,dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device) # Prewitt operator
    ave_filter=torch.tensor([[0.25,0.25],[0.25,0.25]],dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device) # kernal of mean filtering
    down_step=2 # step for down sampling
    hy=hx.transpose(2,3)

    dis_img=dis_img.float().to(device)
    ref_img=ref_img.float().to(device)

    # mean filtering
    ave_dis=F.conv2d(dis_img,ave_filter,stride=1)
    ave_ref=F.conv2d(ref_img,ave_filter,stride=1)
    # downsampling
    ave_dis_down=ave_dis[:,:,0::down_step,0::down_step]
    ave_ref_down=ave_ref[:,:,0::down_step,0::down_step]
    # calculate middle variables like mr, md and so on
    mr_sq=F.conv2d(ave_ref_down,hx)**2+F.conv2d(ave_ref_down,hy)**2
    md_sq=F.conv2d(ave_dis_down,hx)**2+F.conv2d(ave_dis_down,hy)**2
    mr=torch.sqrt(mr_sq)
    md=torch.sqrt(md_sq)
    gms=(2*mr*md+c)/(mr_sq+md_sq+c)
    return torch.std(gms.view(-1))


def cal_wed_ssim_results():
    total_ssim_list = []
    for i in range(1, 4745):
        img_id = ('00000' + str(i))[-5:]
        ssim_list = []
        path1 = './exploration_database_and_code/pristine_images/' + img_id + '.bmp'
        path_pre = './exploration_database_and_code/distort_images/'
        for dist in ['gblur', 'gnoise', 'jpg', 'jp2k']:
            for level in range(1,6):
                path2 = path_pre + dist + '/' + dist + '_' + img_id + '_' + str(level)
                path2 += '.bmp' if dist in ['gblur', 'gnoise'] else '.jpg'
                ssim_list.append(cal_ssim(cv2.imread(path1), cv2.imread(path2)))
        total_ssim_list.append(ssim_list)
    np.save('./results/wed_ssim.npy', np.array(total_ssim_list))


def cal_wed_fsim_results():
    start_time = time.time()
    total_fsim_list = []
    for i in range(1, 4745):
        img_id = ('00000' + str(i))[-5:]
        fsim_list = []
        path1 = './exploration_database_and_code/pristine_images/' + img_id + '.bmp'
        path_pre = './exploration_database_and_code/distort_images/'
        for dist in ['gblur', 'gnoise', 'jpg', 'jp2k']:
            for level in range(1,6):
                path2 = path_pre + dist + '/' + dist + '_' + img_id + '_' + str(level)
                path2 += '.bmp' if dist in ['gblur', 'gnoise'] else '.jpg'
                fsim_list.append(cal_fsim(cv2.imread(path1), cv2.imread(path2)))
        total_fsim_list.append(fsim_list)
    np.save('./results/wed_fsim.npy', np.array(total_fsim_list))


def cal_wed_gmsd_results():
    start_time = time.time()
    total_gmsd_list = []
    for i in range(1, 4745):
        img_id = ('00000' + str(i))[-5:]
        fsim_list = []
        path1 = './exploration_database_and_code/pristine_images/' + img_id + '.bmp'
        path_pre = './exploration_database_and_code/distort_images/'
        for dist in ['gblur', 'gnoise', 'jpg', 'jp2k']:
            for level in range(1,6):
                path2 = path_pre + dist + '/' + dist + '_' + img_id + '_' + str(level)
                path2 += '.bmp' if dist in ['gblur', 'gnoise'] else '.jpg'
                fsim_list.append(cal_gmsd(skimage.io.imread(path2, as_gray=True), skimage.io.imread(path1, as_gray=True)))
        total_gmsd_list.append(fsim_list)
    np.save('./results/wed_gmsd.npy', torch.tensor(total_gmsd_list).cpu().numpy())


if __name__ == '__main__':
    cal_wed_ssim_results()
    cal_wed_fsim_results()
    cal_wed_gmsd_results()

