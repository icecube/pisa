# authors: P.Eller & S. Wren (pde3@psu.edu)
# date:   March 2017


import os

import numpy as np

from pisa.utils.resources import find_resource


class CPUWeight(object):
    """
    Same as GPUWeight but on a CPU

    """

    def __init__(self):
        # TODO: path containing $PISA can be invalid! Use resources & relative
        # package paths instead!

        # compile
        include_dirs = [
            os.path.abspath(find_resource('../utils'))
        ]
        # These parameters are obtained from fits to the paper of Barr
        # E dependent ratios, max differences per flavor (Fig.7)
        self.e1max_mu = 3.
        self.e2max_mu = 43
        self.e1max_e  = 2.5
        self.e2max_e  = 10
        self.e1max_mu_e = 0.62
        self.e2max_mu_e = 11.45
        # Evaluated at
        self.x1e = 0.5
        self.x2e = 3.

        # Zenith dependent amplitude, max differences per flavor (Fig. 9)
        self.z1max_mu = 0.6
        self.z2max_mu = 5.
        self.z1max_e  = 0.3
        self.z2max_e  = 5.
        self.nue_cutoff  = 650.
        self.numu_cutoff = 1000.
        # Evaluated at
        self.x1z = 0.5
        self.x2z = 2.

    def spectral_index_scale(self, true_energy, energy_pivot, delta_index):
        scale = np.power((true_energy/energy_pivot), delta_index)
        return scale

    def apply_ratio_scale(self, flux1, flux2, ratio_scale, sum_const):
        # Normalisation preserving
        if sum_const:
            orig_ratio = flux1/flux2
            orig_sum = flux1 + flux2
            scaled_flux2 = orig_sum / (1 + ratio_scale * orig_ratio)
            scaled_flux1 = ratio_scale * orig_ratio * scaled_flux2
        # Or not
        else:
            scaled_flux1 = ratio_scale * flux1
            scaled_flux2 = flux2
        return scaled_flux1, scaled_flux2

    def LogLogParam(self, energy, y1, y2, x1, x2, use_cutoff, cutoff_value):
        # Philipp took this from OscFit. See code in GPUWeight for relevant
        # comments.
        nu_nubar = np.sign(y2)
        if nu_nubar == 0.0:
            nu_nubar = 1
        y1 = np.sign(y1) * np.log10(np.absolute(y1)+0.0001)
        y2 = np.log10(np.absolute(y2+0.0001))
        modification = nu_nubar * np.power(
            10.0,
            (((y2-y1)/(x2-x1))*(np.log10(energy)-x1)+y1-2.0)
        )
        if use_cutoff:
            modification *= np.exp(-1.0*energy/cutoff_value)
        return modification

    def norm_fcn(self, x, A, sigma):
        # Philipp took this from OscFit. See code in GPUWeight for relevant
        # comments.
        val = A / np.sqrt(2.0 * np.pi * np.power(sigma,2)) * \
            np.exp(-1.0 * np.power(x,2) / (2.0 * np.power(sigma,2)))
        return val

    def shape(self, x):
        # Philipp took this from OscFit. See code in GPUWeight for relevant
        # comments.
        return np.cos(x * np.pi)

    def ModNuMuFlux(self, energy, czenith, e1, e2, z1, z2):
        # Philipp took this from OscFit. See code in GPUWeight for relevant
        # comments.
        A_ave = self.LogLogParam(
            energy=energy,
            y1=self.e1max_mu*e1,
            y2=self.e2max_mu*e2,
            x1=self.x1e,
            x2=self.x2e,
            use_cutoff=False,
            cutoff_value=0
        )
        A_shape = 2.5 * self.LogLogParam(
            energy=energy,
            y1=self.z1max_mu*z1,
            y2=self.z2max_mu*z2,
            x1=self.x1z,
            x2=self.x2z,
            use_cutoff=True,
            cutoff_value=self.numu_cutoff
        )
        val = A_ave - (self.norm_fcn(
            x=czenith,
            A=A_shape,
            sigma=0.32
        ) - 0.75 * A_shape)
        return val

    def ModNuEFlux(self, energy, czenith, e1mu, e2mu, z1mu,
                   z2mu, e1e, e2e, z1e, z2e):
        # Philipp took this from OscFit. See code in GPUWeight for relevant
        # comments.
        A_ave = self.LogLogParam(
            energy=energy,
            y1=self.e1max_mu*e1mu+self.e1max_e*e1e,
            y2=self.e2max_mu*e2mu+self.e2max_e*e2e,
            x1=self.x1e,
            x2=self.x2e,
            use_cutoff=False,
            cutoff_value=0
        )
        A_shape = 1.0 * self.LogLogParam(
            energy=energy,
            y1=self.z1max_mu*z1mu+self.z1max_e*z1e,
            y2=self.z2max_mu*z2mu+self.z2max_e*z2e,
            x1=self.x1z,
            x2=self.x2z,
            use_cutoff=True,
            cutoff_value=self.nue_cutoff
        )
        val = A_ave - (1.5 * self.norm_fcn(
            x=czenith,
            A=A_shape,
            sigma=0.4
        ) - 0.7 * A_shape)
        return val

    def modRatioUpHor(self, kFlav, true_energy, true_coszen, uphor):
        # Philipp took this from OscFit. See code in GPUWeight for relevant
        # comments.
        if kFlav == 0:
            A_shape = 1.0 * np.absolute(uphor) * self.LogLogParam(
                energy=true_energy,
                y1=self.z1max_e + self.z1max_mu,
                y2=self.z2max_e + self.z2max_mu,
                x1=self.x1z,
                x2=self.x2z,
                use_cutoff=True,
                cutoff_value=self.nue_cutoff
            )
        elif kFlav == 1:
            A_shape = 1.0 * np.absolute(uphor) * self.LogLogParam(
                energy=true_energy,
                y1=self.z1max_mu,
                y2=self.z2max_mu,
                x1=self.x1z,
                x2=self.x2z,
                use_cutoff=True,
                cutoff_value=self.numu_cutoff
            )
        else:
            raise ValueError("I got the flavour %i which I don't understand." 
                             " Expect 0 or 1."%kFlav)
        val = 1.0 - 3.5 * np.sign(uphor) * self.norm_fcn(
            x=true_coszen,
            A=A_shape,
            sigma=0.35
        )
        return val

    def modRatioNuBar(self, kNuBar, kFlav, true_e, true_cz,
                      nu_nubar, nubar_sys):
        # Philipp took this from OscFit. See code in GPUWeight for relevant
        # comments.
        if kFlav == 0:
            modfactor = nubar_sys * self.ModNuEFlux(
                energy=true_e,
                czenith=true_cz,
                e1mu=1.0,
                e2mu=1.0,
                z1mu=1.0,
                z2mu=1.0,
                e1e=1.0,
                e2e=1.0,
                z1e=1.0,
                z2e=1.0
            )
        elif kFlav == 1:
            modfactor = nubar_sys * self.ModNuMuFlux(
                energy=true_e,
                czenith=true_cz,
                e1=1.0,
                e2=1.0,
                z1=1.0,
                z2=1.0
            )
        else:
            raise ValueError("I got the flavour %i which I don't understand." 
                             " Expect 0 or 1."%kFlav)
        d0a = np.zeros_like(modfactor)
        d1a = np.ones_like(modfactor)
        if kNuBar < 0:
            return np.maximum(
                d0a,
                d1a/(1.0+(1.0 - nu_nubar)*modfactor)
            )
        elif kNuBar > 0:
            return np.maximum(
                d0a,
                1.0+0.5*modfactor
            )
        else:
            raise ValueError("I got the nu/nubar number of %.2f which I don't"
                             " understand. Should be non-zero"%kNuBar)
                
    def calc_flux(self, n_evts, weighted_aeff, true_energy, true_coszen,
                neutrino_nue_flux, neutrino_numu_flux,
                neutrino_oppo_nue_flux, neutrino_oppo_numu_flux,
                scaled_nue_flux, scaled_numu_flux,
                scaled_nue_flux_shape, scaled_numu_flux_shape,
                nue_numu_ratio, nu_nubar_ratio, kNuBar, delta_index,
                Barr_uphor_ratio, Barr_nu_nubar_ratio,
                true_e_scale,
                **kwargs):

        #TODO This function's arguments need some thought. Currently, it is designed such that it has the same
        #args as the GPU version, even where those args are unused here. It is not really clear though what
        #variables are outputs and what are local, and care must be take to use e.g. np.copyto to ensure changes
        #propagate byond the scope fo this function

        # Delta index systematic
        # 24.0900951261 was hard coded here as the pivot. I don't know why!
        idx_scale = self.spectral_index_scale(
            true_energy=true_energy * true_e_scale,
            energy_pivot=24.0900951261,
            delta_index=delta_index
        )
        # nue/numu ratio systematic
        new_nue_flux, new_numu_flux = self.apply_ratio_scale(
            flux1=neutrino_nue_flux,
            flux2=neutrino_numu_flux,
            ratio_scale=nue_numu_ratio,
            sum_const=True
        )
        new_nue_oppo_flux, new_numu_oppo_flux = self.apply_ratio_scale(
            flux1=neutrino_oppo_nue_flux,
            flux2=neutrino_oppo_numu_flux,
            ratio_scale=nue_numu_ratio,
            sum_const=True
        )
        # nu/nubar ratio systematic
        if kNuBar < 0:
            new_nue_oppo_flux2, new_nue_flux2 = self.apply_ratio_scale(
                flux1=neutrino_oppo_nue_flux,
                flux2=neutrino_nue_flux,
                ratio_scale=nu_nubar_ratio,
                sum_const=True
            )
            new_numu_oppo_flux2, new_numu_flux2 = self.apply_ratio_scale(
                flux1=neutrino_oppo_numu_flux,
                flux2=neutrino_numu_flux,
                ratio_scale=nu_nubar_ratio,
                sum_const=True
            )
        else:
            new_nue_flux2, new_nue_oppo_flux2 = self.apply_ratio_scale(
                flux1=neutrino_nue_flux,
                flux2=neutrino_oppo_nue_flux,
                ratio_scale=nu_nubar_ratio,
                sum_const=True
            )
            new_numu_flux2, new_numu_oppo_flux2 = self.apply_ratio_scale(
                flux1=neutrino_numu_flux,
                flux2=neutrino_oppo_numu_flux,
                ratio_scale=nu_nubar_ratio,
                sum_const=True
            )
        # Barr flux
        new_nue_flux2 *= self.modRatioNuBar(
            kNuBar=kNuBar,
            kFlav=0,
            true_e=true_energy * true_e_scale,
            true_cz=true_coszen,
            nu_nubar=1.0,
            nubar_sys=Barr_nu_nubar_ratio
        )
        new_numu_flux2 *= self.modRatioNuBar(
            kNuBar=kNuBar,
            kFlav=1,
            true_e=true_energy * true_e_scale,
            true_cz=true_coszen,
            nu_nubar=1.0,
            nubar_sys=Barr_nu_nubar_ratio
        )
        np.copyto(scaled_nue_flux,new_nue_flux2) #Use copyto to fill np array from another (using '=' sets the pointer reference without changing the underlying object)
        np.copyto(scaled_numu_flux,new_numu_flux2)
        np.copyto(scaled_nue_flux_shape,
            new_nue_flux2 * idx_scale * self.modRatioUpHor(
                kFlav=0,
                true_energy=true_energy * true_e_scale,
                true_coszen=true_coszen,
                uphor=Barr_uphor_ratio
            )
        )
        np.copyto(scaled_numu_flux_shape,
            new_numu_flux2 * idx_scale * self.modRatioUpHor(
                kFlav=1,
                true_energy=true_energy * true_e_scale,
                true_coszen=true_coszen,
                uphor=Barr_uphor_ratio
            )
        )


    def calc_weight(self, n_evts, weighted_aeff, true_energy, true_coszen,
                    scaled_nue_flux_shape, scaled_numu_flux_shape,
                    nue_flux_norm, numu_flux_norm,
                    linear_fit_MaCCQE, quad_fit_MaCCQE,
                    linear_fit_MaCCRES, quad_fit_MaCCRES,
                    prob_e, prob_mu, pid, weight,
                    livetime, aeff_scale,
                    Genie_Ma_QE, Genie_Ma_RES,
                    true_e_scale,
                    **kwargs):
        nue_flux = scaled_nue_flux_shape * nue_flux_norm
        numu_flux = scaled_numu_flux_shape * numu_flux_norm
        # GENIE axial mass systemtic
        aeff_QE = 1.0 + quad_fit_MaCCQE * np.power(
            Genie_Ma_QE,2
        ) + linear_fit_MaCCQE * Genie_Ma_QE
        aeff_RES = 1.0 + quad_fit_MaCCRES * np.power(
            Genie_Ma_RES,2
        ) + linear_fit_MaCCRES * Genie_Ma_RES
        # Calculate weight
        np.copyto(weight, 
            aeff_scale * livetime * 
            weighted_aeff * aeff_QE * 
            aeff_RES * 
            ( (nue_flux * prob_e) + (numu_flux * prob_mu) )
        )
        

    def calc_sum(self, n_evts, x, out):
        x[0] = np.sum(x)


    def calc_sumw2(self, weight, sumw2, **kwargs):
        sumw2 = weight * weight #TODO Check this

