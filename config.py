optimization = dict(
    corruption = 'random_inpaint',        #inpaint, random_inpaint, super-res, denoising, colorization
    steps = [300,200,100,100],
    lr = 0.1,
    input_file = 'obama.jpg',
    output_file = 'output.jpg',
    mse_coeff = 1,
    lpips_coeff = 1,
    geocross_coeff = 0.01,
    super_res=2,
    random_inpainting_percentage=0.90,
    noise_std = 100
)