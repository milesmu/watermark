#!python3
# --encoding: utf8 --
from src import *

# make sure all the images must be same size
folder = "./images/fotolia"
gx, gy, gxlist, gylist = estimate_watermark(folder)

# est = poisson_reconstruct(gx, gy, np.zeros(gx.shape)[:,:,0])
cropped_gx, cropped_gy = crop_watermark(gx, gy)
W_m = poisson_reconstruct(cropped_gx, cropped_gy)

# random photo
sample = "images/fotolia/fotolia_137840668.jpg"
img = cv2.imread(sample)

# TODO: images size must be same
# img = cv2.resize(img, (512,768), interpolation = cv2.INTER_AREA)
im, start, end = watermark_detector(img, cropped_gx, cropped_gy)

print("start: %s, end: %s\n" % (start, end))
plt.imshow(im)
plt.show()

# We are done with watermark estimation
# W_m is the cropped watermark
num_images = len(gxlist)

# assume the wartermarks with the same size and  position
J, img_paths = get_cropped_images(folder, num_images, start, end, cropped_gx.shape)

# Wm = (255*PlotImage(W_m))
Wm = W_m - W_m.min()

# get threshold of W_m for alpha matte estimate
alph_est = estimate_normalized_alpha(J, Wm, len(J))
alph = np.stack([alph_est, alph_est, alph_est], axis=2)
C, est_Ik = estimate_blend_factor(J, Wm, alph)

alpha = alph.copy()
for i in range(3):
	alpha[:,:,i] = C[i]*alpha[:,:,i]

Wm = Wm + alpha*est_Ik

W = Wm.copy()
for i in range(3):
	W[:,:,i]/=C[i]

# now we have the values of alpha, Wm, J
print("Solve for all images...")
Wk, Ik, W, alpha1 = solve_images(J, W_m, alpha, W)

print("TODO: patch the image with inpainted area. ;-)")
cv2.imwrite("inpainted.jpg", Ik[0])

#W_m_threshold = (255*PlotImage(np.average(W_m, axis=2))).astype(np.uint8)
#ret, thr = cv2.threshold(W_m_threshold, 127, 255, cv2.THRESH_BINARY)  

print("done!")
