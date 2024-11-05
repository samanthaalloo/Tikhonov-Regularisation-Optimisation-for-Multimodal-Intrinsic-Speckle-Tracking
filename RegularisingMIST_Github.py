# ---------------------------------------------------------------------------------
# Written by Samantha Alloo -- 9/10/2024

# This code performs the 2023 MIST algorithm published in Alloo, S. J., Morgan, K. S., Paganin, D. M., &
# Pavlov, K. M. (2023). Multimodal intrinsic speckle-tracking (MIST) to extract images of rapidly-varying
# diffuse X-ray dark-field. Scientific Reports, 13(1), 5424, however, the value of the regularisation parameter in the
# inverse Laplacian operator used to recover the phase is not required. Here, we present an algorithm that iteratively
# refines this value of this Tikhonov-regularisation parameter by employing the TIE-based algorithm developed in
# Pavlov, K. M., Li, H., Paganin, D. M., Berujon, S., Rougé-Labriet, H., & Brun, E. (2020). Single-shot x-ray
# speckle-based imaging of a single-material object. Physical Review Applied, 13(5), 054023.
# ---------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import scipy
import csv
from scipy import ndimage, misc
from PIL import Image
import time
from scipy.ndimage import median_filter, gaussian_filter
import fabio
import pyedflib
import h5py
import colorsys
import time
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
# # ---------------------------------------------------------------------------------
# # ---------------------------------------------------------------------------------
# Test Data: Four-rod sample imaged at the MicroCT beamline at the Australian synchrotron
data = r'C:\Users\sall0037\Documents\Experimental_Data\5_MaskComparison_MCT 19663\4WoodSample_MCTApril23' # Directory where the data is located
os.chdir(data)

num_masks = 13
gamma = 2335 # ratio of delta to beta: taken as that of PMMA at 25 keV
wavelength = 4.9594*10**-5 # [microns]
prop = 0.7*10**6 # [microns]
pixel_size = 6.5 # [microns]


savedir = os.path.join(data, "RetrievedSignals") # Combine the base directory and new folder name to create the full path
save = os.makedirs(savedir, exist_ok=True) # Create the new folder (if it doesn't already exist) for the ouputs of this script to be saved to

ff = np.double(np.asarray(Image.open('FF_1m.tif')))[870:1300,140:2360]
rows, columns = ff.shape

Ir = np.empty([int(num_masks),int(rows),int(columns)])
Is = np.empty([int(num_masks),int(rows),int(columns)])
dc = 0
for k in range(1,int(num_masks+1)):
        i = str(k)
        # -------------------------------------------------------------------------
        # Reading in data: change string for start of filename as required
        ir = np.double(np.asarray(Image.open('Ref{}.tif'.format(str(i)))))[870:1300,140:2360]
        isa = np.double(np.asarray(Image.open('Sam{}.tif'.format(str(i)))))[870:1300,140:2360]

        ir = (ir-dc)/(ff-dc)
        isa = (isa-dc)/(ff-dc)

        Is[int(k-1)] = (isa)  # shape =  [num_masks, rows, columns]
        Ir[int(k-1)] = (ir)  # shape =  [num_masks, rows, columns]

        print('Completed Reading Data From Mask = ' + str(i))
        # -------------------------------------------------------------------------
# # ---------------------------------------------------------------------------------
# # ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# Step 1) Recover the ground-truth reconstructions
start_time = time.time() # Starting the time so the computation time is printed at the end
def kspace_kykx(image_shape: tuple, pixel_size: float = 1):
    # Multiply by 2pi for correct values, since DFT has 2pi in exponent
    rows = image_shape[0]
    columns = image_shape[1]
    ky = 2*math.pi*np.fft.fftfreq(rows, d=pixel_size) # spatial frequencies relating to "rows" in real space
    kx = 2*math.pi*np.fft.fftfreq(columns, d=pixel_size) # spatial frequencies relating to "columns" in real space
    return ky, kx
def PavlovTransmission(Is, Ir, pixel_size,gamma, prop, wavelength):
    # ---------------------------------------------------------------
    # Definitions:
    # Is: One sample-plus-speckle image [ndarray]
    # Ir: One speckle-only image [ndarray]
    # pixel_size: Pixel size of the detector [microns]
    # gamma: Ratio of delta to beta for the object
    # prop: Distance between the sample and detector [microns]
    # wavelength: Wavelength of the monochromatic X-ray beam used during imaging [microns]
    # ---------------------------------------------------------------

    IsIr_mirror = np.concatenate((Is/Ir, np.fliplr(Is/Ir)), axis=1) # Doing mirroring (horizontally and vertically) to enforce periodicity for DFT implementation
    IsIr_mirror = np.concatenate((IsIr_mirror, np.flipud(IsIr_mirror)), axis=0)

    ft_IsIr = np.fft.fft2(IsIr_mirror) # Taking the 2D Fourier Transform
    ky, kx = kspace_kykx(ft_IsIr.shape, pixel_size) # Finding the Fourier-space spatial frequencies
    ky2kx2 = np.add.outer(ky**2,kx**2) # Making the k_x^2 + k_y^2 term with correct dimensions
    ins_ifft = ft_IsIr/(1 +((prop*gamma*wavelength)/(4*math.pi))*ky2kx2) # This is what needs the inverse Fourier transform applied to recover the attenuation term
    Iob = np.real(np.fft.ifft2(ins_ifft)) # Taking the inverse Fourier transform and only look at the real component to give the transmission term
    Iob_crop = Iob[:Is.shape[0], :Is.shape[1]] # Cropping off the mirror done

    Phase_crop = gamma/2 * np.log(Iob_crop) # This is the object's recovered phase, using the projection approximation

    return Iob_crop, Phase_crop
os.chdir(savedir)
I_TIE, phase_TIE = PavlovTransmission(Is[0], Ir[0], pixel_size,gamma, prop, wavelength) # These are the ground truth images that are used to optmimize Alloo's MIST phase retrieval
Image.fromarray(I_TIE).save('Pavlov2020_Iob_{}.tiff'.format(str(gamma)))
Image.fromarray(phase_TIE).save('Pavlov2020_Phase_{}.tiff'.format(str(gamma)))
# ---------------------------------------------------------------------------------
# Step 2) Recover a first 'guess' for the phase-retrieved image using the MIST algorithm
def lowpass_2D(image, r, pixel_size):
    # -------------------------------------------------------------------
    # This function will generate a low-pass filter and suppress the input spatial frequencies, kr of the image, beyond some defined
    # spatial frequency r
    # DEFINITIONS
    # image: input image whos spatial frequencies you want to suppress
    # r: spatial frequency you want to suppress beyond [pixel number]
    # pixel_size: physical size of pixel [microns]
    # -------------------------------------------------------------------
    rows = image.shape[0]
    columns = image.shape[1]
    m = np.fft.fftfreq(rows, d=pixel_size)  # spatial frequencies relating to "rows" in real space
    n = np.fft.fftfreq(columns, d=pixel_size)  # spatial frequencies relating to "columns" in real space
    ky = (2 * math.pi * m)  # defined by row direction
    kx = (2 * math.pi * n)  # defined by column direction

    kx2 = kx ** 2
    ky2 = ky ** 2
    kr2 = np.add.outer(ky2, kx2)
    kr = np.sqrt(kr2)

    lowpass_2d = np.exp(-r * (kr ** 2))

    # plt.imshow(lowpass_2d)
    # plt.title('Low-Pass Filter 2D')
    # plt.colorbar()
    # plt.show()

    return lowpass_2d
def highpass_2D(image, r, pixel_size):
    # -------------------------------------------------------------------
    # This function will generate a high-pass filter and suppress the input spatial frequencies, kr of the image, up to some defined
    # spatial frequency r
    # DEFINITIONS
    # image: input image whos spatial frequencies you want to suppress
    # r: spatial frequency you want to suppress beyond [pixel number]
    # pixel_size: physical size of pixel [microns]
    # -------------------------------------------------------------------
    rows = image.shape[0]
    columns = image.shape[1]
    m = np.fft.fftfreq(rows, d=pixel_size)  # spatial frequencies relating to "rows" in real space
    n = np.fft.fftfreq(columns, d=pixel_size)  # spatial frequencies relating to "columns" in real space
    ky = (2 * math.pi * m)  # defined by row direction
    kx = (2 * math.pi * n)  # defined by column direction

    kx2 = kx ** 2
    ky2 = ky ** 2
    kr2 = np.add.outer(ky2, kx2)
    kr = np.sqrt(kr2)

    highpass_2d = 1 - np.exp(-r * (kr ** 2))

    # plt.imshow(highpass_2d)
    # plt.title('High-Pass Filter 2D')
    # plt.colorbar()
    # plt.show()

    return highpass_2d
def midpass_2D(image, r, pixel_size):
    # -------------------------------------------------------------------
    # This function will generate a low-pass filter and suppress the input spatial frequencies, kr of the image, up to some defined
    # spatial frequency r
    # DEFINITIONS
    # image: input image whos spatial frequencies you want to suppress
    # r: spatial frequency you want to suppress beyond [pixel number]
    # pixel_size: physical size of pixel [microns]
    # -------------------------------------------------------------------
    rows = image.shape[0]
    columns = image.shape[1]
    m = np.fft.fftfreq(rows, d=pixel_size)  # spatial frequencies relating to "rows" in real space
    n = np.fft.fftfreq(columns, d=pixel_size)  # spatial frequencies relating to "columns" in real space
    ky = (2 * math.pi * m)  # defined by row direction
    kx = (2 * math.pi * n)  # defined by column direction

    kx2 = kx ** 2
    ky2 = ky ** 2
    kr2 = np.add.outer(ky2, kx2)
    kr = np.sqrt(kr2)

    highpass_2d = 1 - np.exp(-r * (kr ** 2))

    C = np.zeros(columns, dtype=np.complex128)
    C = C + 0 + 1j
    ikx = kx * C  # (i) * spatial frequencies in x direction (along columns) - as complex numbers ( has "0" in the real components, and "kx" in the complex)
    denom = np.add.outer((-1 * ky), ikx)  # array with ikx - ky (DENOMINATOR)

    midpass_2d = np.divide(complex(1., 0.) * highpass_2d, denom, out=np.zeros_like(complex(1., 0.) * highpass_2d),
                           where=denom != 0)  # Setting output equal to zero where denominator equals zero


    return midpass_2d

coeff_D = []  # Empty lists to store terms required to solve the system of linear equations
coeff_dx = []
coeff_dy = []
lapacaian = []
RHS = []

coefficient_A = np.empty([int((num_masks)), 4, int(rows),
                          int(columns)])  # Empty arrays to put calculated terms in and to perform QR decomposition on
coefficient_b = np.empty([int((num_masks)), 1, int(rows), int(columns)])

for i in range(num_masks):  # This forloop will calculate and store all of the requires coefficients for the system of linear equations
    rhs = (1 / prop) * (Ir[i, :, :] - Is[i, :, :])
    lap = Ir[i, :, :]
    deff = (-1) * np.divide(ndimage.laplace(Ir[i, :, :]), pixel_size ** 2)
    dy, dx = np.gradient(Ir[i, :, :], pixel_size)
    dy_r = -2 * dy
    dx_r = -2 * dx

    coeff_D.append(deff)
    coeff_dx.append(dx_r)
    coeff_dy.append(dy_r)
    lapacaian.append(lap)
    RHS.append(rhs)

# Establishing the system of linear equations: Ax = b where x = [Laplacian(1/wavenumber*Phi - D), D, dx, dy]
for n in range(len(coeff_dx)):
    coefficient_A[n, :, :, :] = np.array([lapacaian[n], coeff_D[n], coeff_dx[n], coeff_dy[n]])
    coefficient_b[n, :, :, :] = RHS[n]

identity = np.identity(4)  # This is applying the Tikhonov Regularisation to the QR decomposition
alpha = np.std(coefficient_A) / 10000  # This is the optimal Tikhonov regularisation parameter (may need tweaking if the system is overly unstable)
reg = np.multiply(alpha, identity)  # 4x4 matrix representing the Tikhinov regularization on the coefficient array
reg_repeat = np.repeat(reg, rows * columns).reshape(4, 4, rows,
                                                    columns)  # Repeating the regularisation across all pixel positions
zero_repeat = np.zeros(
    (4, 1, rows, columns))  # 4x1 matrix representing the Tikhinov regularization on the righthand-side vector
coefficient_A_reg = np.vstack(
    [coefficient_A, reg_repeat])  # Coefficient matrix of linear system that is Tikhonov regularised
coefficient_b_reg = np.vstack([coefficient_b, zero_repeat])  # RHS of linear system that is Tikhonov regularised

reg_Qr, reg_Rr = np.linalg.qr(coefficient_A_reg.transpose([2, 3, 0, 1]))
# Now here, we just use a solver to solve Rx = Q^tb instead of taking the inverse - Chris 27.06.2023
reg_x = np.linalg.solve(reg_Rr, np.matmul(np.matrix.transpose(reg_Qr.transpose([2, 3, 1, 0])),
                                          coefficient_b_reg.transpose([2, 3, 0, 1])))

lap_phiDF = reg_x[:, :, 0, 0]  # Laplacian term array (Laplacian(1/wavenumber*Phi - D))
DFqr = (reg_x[:, :, 1, 0]) / prop  # DF array (DF_reg)
dxDF = (reg_x[:, :, 2, 0]) / prop  # d(DF)/dx array
dyDF = (reg_x[:, :, 3, 0]) / prop  # d(DF)/dy array

os.chdir(savedir)
# Uncomment the below if you want to save the solutions for de-bugging
# DFphiim = Image.fromarray(lap_phiDF).save(
#     'Evolve_LapDFPhiphase_{}.tif'.format(
#         'mask' + str(num_masks) + 'e' + str(alpha)))  # Saving solutions of the system of linear equations
# DFim = Image.fromarray(DFqr).save('Evolve_regDFphase_{}.tif'.format('mask' + str(num_masks) + 'e' + str(alpha)))
# dxDFim = Image.fromarray(dxDF).save(
#     'Evolve_dxDFphase_{}.tif'.format('mask' + str(num_masks) + 'e' + str(alpha)))
# dyDFim = Image.fromarray(dyDF).save(
#     'Evolve_dyDFphase_{}.tif'.format('mask' + str(num_masks) + 'e' + str(alpha)))
#

# Determing TRUE dark-field signal by aggregating the solutoins
cutoff = 10

i_dyDF = dyDF * (np.zeros((DFqr.shape),
                          dtype=np.complex128) + 0 + 1j)  # (i) * derivative along rows of DF, has "0" in the real components, and "d(DF)/dx" in the complex

insideft = dxDF + i_dyDF

insideftm = np.concatenate((insideft, np.flipud(insideft)), axis=0)
insideftm = np.concatenate((insideftm, np.fliplr(insideftm)), axis=1) # Mirroring the term inside the Fourier transform to enforce periodic boundary conditions

ft_dx_idy = np.fft.fft2(insideftm)
MP = midpass_2D(ft_dx_idy, cutoff, pixel_size)
MP_deriv = MP * ft_dx_idy  # This is the 'derivative solution' mid-pass filtered

DFqrm = np.concatenate((DFqr, np.flipud(DFqr)), axis=0)
DFqrm = np.concatenate((DFqrm, np.fliplr(DFqrm)), axis=1) # Mirroring the term inside the Fourier transform to enforce periodic boundary conditions

ft_DFqr = np.fft.fft2(DFqrm)
LP = lowpass_2D(ft_DFqr, cutoff, pixel_size)
LP_DFqr = LP * ft_DFqr  # This is the QR derived solution low-pass filtered

combined = LP_DFqr + MP_deriv  # Combining the two solutions (note, two filters sum to 1)

DF_filtered = np.real((np.fft.ifft2(combined)))[0:int(rows), 0:int(columns)]  # Inverting Fourier transform to calculate the TRUE dark-field and then cropping off the mirroring

DFFim = Image.fromarray(np.real(DF_filtered)).save(
    'Evolve_DFphase_Agg_{}.tif'.format('mask' + str(num_masks) + 'e' + str(alpha) + 'r' + str(cutoff)))

# Calculating the phase-shifts and attenuation term - could maybe improve by calcualting the Iob for all sets and averaing: this is where we need to iterate
ref = Ir[0, :, :] # We just need a single reference speckle and single sample-speckle image for this reconstruction
sam = Is[0, :, :]

lapphi = (ref - sam + prop ** 2 * np.divide(ndimage.laplace(DF_filtered * ref), pixel_size ** 2)) * (
        (2 * math.pi) / (wavelength * prop * ref)) # This is the Laplacian of the phase

# Applying the inverse transverse Laplacian operator
lapphi_m = np.concatenate((lapphi, np.flipud(lapphi)), axis=0) # Doing mirroring to enforce periodicity
lapphi_m = np.concatenate((lapphi_m, np.fliplr(lapphi_m)), axis=1)

ky, kx = kspace_kykx(lapphi_m.shape, pixel_size) # Finding fourier-space frequencies
kr2 = np.add.outer(ky**2, kx**2)

ftimage = np.fft.fft2(lapphi_m)

# Step 3) Perform the iterative algorithm to optimize the value of \epsilon required to recover the phase image in the Alloo et al. 2023 MIST approach
epsilon_inital = kr2[kr2 > 0][0] if np.any(kr2 > 0) else None # we take the first non-zero Fourier-Space frequency as the first epsilon guess

# Using the mean square error as the metric we want to minimise
ep_value = []
MSE_value = []
def EpsilonOpt_IterativeAlgorithm_MSE(epsilon_initial, phase_TIE, kr2, ftimage):
    # ----------------------------------------------------------------------
    # Optimizes the epsilon value by minimizing the Mean Square Error (MSE)
    # between phase reconstructions from the Alloo et al. 2023 MIST approach and
    # Pavlov et al. 2020 TIE approach.
    # ----------------------------------------------------------------------
    # epsilon_initial: the first guess epsilon value
    # phase_TIE: ground truth phase reconstruction from Pavlov et al. TIE
    # kr2: Fourier space spatial frequencies
    # ftimage: image to which the (1/kr^2) filter is applied.
    # ----------------------------------------------------------------------

    rows, columns = phase_TIE.shape
    iteration = 0

    # Initialize variables to track the lowest MSE and corresponding epsilon
    previous_MSE = float('inf')  # Start with a large value
    best_epsilon = epsilon_initial
    best_phase_FP_it = None

    # --- First Iteration Loop (Find the best initial epsilon) ---
    while True:
        print(f"Iteration {iteration}: Epsilon Value = {epsilon_initial}")

        # 1) Compute the MIST phase reconstruction for the current epsilon value:
        regdiv = 1 / (kr2 + epsilon_initial)  # Regularization division
        phase_FP_it = np.real(
            -1 * np.fft.ifft2(regdiv * ftimage)[0:rows, 0:columns])  # Inverse FFT and filter applied for phase reconstruction

        # The below line will save the phase reconstructions for different magnitude's of \epsilon. Uncomment if you wish to inspect
        #Image.fromarray(phase_FP_it).save('MSE_Phi_e{}.tif'.format(str(epsilon_initial)))

        # 2) Compute the mean square error of the two phase images
        squared_diff = (phase_TIE - phase_FP_it) ** 2  # Compute squared differences between pixel values
        MSE = np.mean(squared_diff, dtype=np.float64)  # Compute the mean of the squared differences
        print(f"MSE at iteration {iteration}: {MSE}")

        # 3) Check if MSE is improving
        if MSE < previous_MSE:
            previous_MSE = MSE
            best_epsilon = epsilon_initial
            best_phase_FP_it = phase_FP_it
        else:
            print(f"MSE increased at iteration {iteration}. Stopping.")
            break  # Stop if MSE starts increasing

        # 4) Update epsilon by multiplying by 10 for the next iteration
        epsilon_initial *= 5 # !! NOTE !! If the algorithm stops after the fist fine-tuning iteration, it means that this step is too large 
            # and the minimum has been overstep. To fix this, simply change this to 2.
        iteration += 1

        ep_value.append(epsilon_initial)
        MSE_value.append(MSE)

    # --- Fine-tuning Search around the Best Epsilon ---
    print(f"Starting fine-tuning search around epsilon = {best_epsilon}")

    epsilon_order = np.floor(np.log10(best_epsilon))  # Get the order of magnitude of previous_epsilon
    epsilon_step = 10 ** (epsilon_order - 3)  # Step size is three orders smaller than previous_epsilon

    best_fine_MSE = previous_MSE  # Start with the previous best MSE
    best_fine_epsilon = best_epsilon  # Start with the previous best epsilon
    best_fine_phase = best_phase_FP_it  # Start with the previous best phase reconstruction

    # Initialize a variable for the previous MSE during fine-tuning
    previous_MSE_fine = float('inf')

    # Initialize epsilon for fine-tuning
    epsilon_fine = 10 ** (epsilon_order)

    while True:
        # Compute the MIST phase reconstruction for the finer epsilon value
        regdiv = 1 / (kr2 + epsilon_fine)
        phase_FP_it_fine = np.real(
            -1 * np.fft.ifft2(regdiv * ftimage)[0:rows, 0:columns])

        # The line below will save each of the fine-tune phase images for different epsilon values. Uncomment if you wish to inspect
        #Image.fromarray(phase_FP_it_fine).save('MSE_FineTune_Phi_e{}.tif'.format(str(epsilon_fine)))

        # Compute the mean square error for the finer epsilon value
        squared_diff_fine = (phase_TIE - phase_FP_it_fine) ** 2
        MSE_fine = np.mean(squared_diff_fine)
        print(f"Fine-tuning MSE for epsilon = {epsilon_fine}: MSE = {MSE_fine}")

        ep_value.append(epsilon_fine)
        MSE_value.append(MSE_fine)

        # Check if MSE is increasing and stop if it is
        if MSE_fine > previous_MSE_fine:
            print(f"MSE increased for epsilon = {epsilon_fine}. Stopping fine-tuning.")
            break  # Stop fine-tuning if MSE starts increasing

        # Update the best fine MSE and epsilon if a better value is found
        previous_MSE_fine = MSE_fine
        best_fine_MSE = MSE_fine
        best_fine_epsilon = epsilon_fine
        best_fine_phase = phase_FP_it_fine

        # Increment epsilon for the next iteration
        epsilon_fine += epsilon_step

    # Return the optimal phase reconstruction and epsilon value
    print(f"Best fine-tuned epsilon: {best_fine_epsilon} with MSE: {best_fine_MSE}")

    # Write trialled epsilon values corresponding MSE values to the CSV file
    values = np.array([ep_value, MSE_value])
    with open('Epsilon&MSE.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(values)

    return best_fine_phase, best_fine_epsilon


optimal_phase_FP_MSE, optimal_epsilon_MSE = EpsilonOpt_IterativeAlgorithm_MSE(epsilon_inital, phase_TIE, kr2, ftimage) # This runs the iterative function defined above
Image.fromarray(optimal_phase_FP_MSE).save('MSE_Optimal_Phi_e{}.tif'.format(str(optimal_epsilon_MSE) + 'gamma' + str(gamma) + 'r' + str(cutoff))) # This saves the optimally retrieved phase image


# Extracting the attenuation term and the attenuation-corrected dark-field comes after finding the optimal inverse transverse Laplacian operator regularization parameter
Iob = np.exp(2 * optimal_phase_FP_MSE / (gamma))  # This is the object's attenuation term
Iob_im = Image.fromarray(Iob).save('Evolve_Iob_e{}.tif'.format(str(optimal_epsilon_MSE) + 'gamma' + str(gamma) + 'r' + str(cutoff)))

DF_atten = np.real(DF_filtered / Iob)  # The object's TRUE attenuating-object approximation of the dark-field

DFattim = Image.fromarray(np.real(DF_atten)).save(
    'DFatten_{}.tif'.format('mask' + str(num_masks) + 'gamma' + str(gamma) + 'r' + str(cutoff)))

# Record the end time
end_time = time.time()

# Calculate and print the time taken
print("Time taken to run the code:", end_time - start_time, "seconds")
