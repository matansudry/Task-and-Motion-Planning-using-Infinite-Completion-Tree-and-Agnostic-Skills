from diffusion.networks.gsc.cond_diffusion import Diffusion
#from diffusion.networks.dec_diffuser.diffusion import GaussianDiffusion

NETWORK_CATALOG = {
    "Diffusion_gsc": Diffusion,
    #"dec_diffusion": GaussianDiffusion,
}