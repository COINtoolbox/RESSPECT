# Copyright 2020 resspect software
# Author: Santiago Gonzalez-Gaitan
#
# created on 18 May 2020
#
# adapted from:
# modified public DECam exposure time calculator by
# CC F. Forster, J. Martinez, J.C. Maureira, 
# https://github.com/fforster/HiTS-public
# 
#
# Licensed GNU General Public License v3.0;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# For more on DECam see: 
# https://www.noao.edu/meetings/decam/media/DECam_Data_Handbook.pdf


__all__ = ['ExpTimeCalc']

import numpy as np
from scipy.interpolate import interp1d

class ExpTimeCalc(object):
    """Class allowing calculation of exposure time.

    Attributes
    ----------
    all_filter_eff: np.array
        Efficiency for all filters.
    all_corrector_eff: np.array
        Optical corrector efficiency for all filters.
    aperture_eff: float
        Aperture efficiency: factor to multiply FWHM with
    CCD_eff_filt: np.array
        CCD efficiency per filter.
    cspeed: float
        Speed of light in nm.
    diameter: float
        Telescope diameter in meters.
    gain: float
        Camera gain.
    hPlanck_MKS: float
        Planck's constant in m2 kg / s.
    magsarray: np.array
        Span of possible limiting magnitudes
    num_atm_eff: np.array
       Atmospheric transmission at each center wavelength of 
       the instrument filters (ugrizY). 
       Default: [0.7,0.8,0.9,0.9,0.9,0.95]
    pixelsize: float
        Pixels size in arc/pix.
    RON_pix: float
        Readout noise per pixel, electrons.
    prim_refl_filt: np.array
        Reflectivity of the primary mirror at each center 
        wavelength of the instrument filters (ugrizY). 
        Default: [0.89,0.89,0.88,0.87,0.88,0.9]
    seeing: dict
        Seeing in different bands (keywords) in arcsec.
    texparray: np.array
        Span of possible exposure times. 
    vig: float
        Vignetting.
    u: float
        Central wavelength for u-band in nm.
    g: float
        Central wavelength for g-band in nm.
    r: float
        Central wavelength for r-band in nm.
    i: float
        Central wavelength for i-band in nm.
    z: float
        Central wavelength for z-band in nm.
    Y: float
        Central wavelength for Y-band in nm.

    Methods
    -------
    findexptime(mag: float, SNRin:float)
        Calculates required exposure time.
    findmag(exptime: float,SNRin:float)
        Calculates magnitude limit.
    FWHM(band: str, airmass: float)
        Computes FWHM.
    SNR(mag: float, exptime: float)
        Computes SNR.
    """
    
    def __init__(self):     
        self.all_filter_eff = np.asarray([0.95,0.9,0.9,0.9,0.9,0.9])
        self.all_corrector_eff = np.asarray([0.75,0.86,0.86,0.86,0.86,0.75])
        self.aperture_eff = 2.04 
        self.CCD_eff_filt = np.asarray([0.25,0.59,0.75,0.85,0.85,0.5])
        self.cspeed = 2.99792458e17     
        self.diameter = 4.0             # in meters  (DEcam default)
        self.gain = 4.0                 # DECam typical gain
        self.hPlanck_MKS = 6.62606957e-34 
        self.magsarray = np.linspace(15., 30., 100000)
        self.num_atm_eff = np.array([0.7,0.8,0.9,0.9,0.9,0.95])
        self.pixelsize = 0.264          # arcsec/pix (DECam default)
        self.prim_refl_filt = np.asarray([0.89,0.89,0.88,0.87,0.88,0.9])
        self.RON_pix = 7.0
        self.seeing = {}                # store all seeing values
        self.seeing['r'] = 0.75         # median value at CTIO
        self.texparray = np.linspace(60., 18000., 100000)  # 60s - 5h
        self.vig = 1.0
        self.u = 375                    # DECam defaults
        self.g = 473.5 
        self.r = 638.5 
        self.i = 775.5 
        self.z = 922.5 
        self.Y = 995   

        # scaling between seeing at different filters
        self.seeing['g'] = self.seeing['r'] * (self.r / self.g)**0.2
        self.seeing['u'] = 0.2 + self.seeing['g'] * (self.g / self.u)**0.2
        self.seeing['i'] = self.seeing['r'] * (self.r / self.i)**0.2
        self.seeing['z'] = self.seeing['i'] * (self.i / self.z)**0.2
        self.seeing['Y'] = self.seeing['z'] * (self.z / self.Y)**0.2

    def FWHM(self, band: str, airmass: float):
        """Computes FWHM.

        Parameters
        ----------
        band: str
            Broad band filter.
        airmass: float
            Airmass.

        Returns
        -------
        FWHM_arcsec: float
            FWHM in arcsec
        """
    
        # select seeing depending on band
        seeing_arcsec = self.seeing[band]
        
        # seeing correction factor due to airmass
        fseeing_airmass = 1. / np.cos(np.arccos(1. / airmass))**(3./5.)

        FWHM_arcsec = np.sqrt((seeing_arcsec * fseeing_airmass)**2 + 0.63**2)
        
        return FWHM_arcsec

    def SNR(self, mag: float, exptime: float, cwl_nm=500.0, bandpass_nm=1.0,
            band='r', airmass=1.25, skymode='ADU', skymag=22.0, nread=1,
            skyADU=120, fwhm=1.0):
        """Computes SNR. 

        Parameters
        ----------
        mag: float
            Magnitude. 
        exptime: float
            Exposure time.
        airmass: float (optional)
            Airmass. Default is 1.25.
        band: str (optional)
            Broad band filter. Default is 'r'.
        bandpass_nm: float (optional)
            Width of wavelength in nm. Default is 1.
        cwl_nm: float (optional)
            Central wavelength in nm. Default is 500.
        fwhm: float (optional)
            FWHM. Default is 1.0.
        nread: int (optional)
            Number of times CCD is readout, i.e. for higher 
            readout rates, this increases the noise. Default: 1.
        skyADU: float (optional)
            Sky brightness in ADU. Default is 120. 
            Only used if 'skymode' \in ['ADU', 'ADU-FWHM']. 
        skymag: float (optional)
            Sky brightness in magnitude. Default is 22.0. 
        skymode: str (optional)
            If 'mag': sky given in mag per arcsec2 at zenith (skymag)
                    FWHM derived from seeing in r band, band and airmass
            If 'ADU': use empirical sky in ADU (skyADU), FWHM derived from
                    seeing in r band, band and airmass
            If 'mag-FWHM': sky given in mag per arcsec2 at zenith (skymag),
                         use empirical FWHM (fwhm)
            If 'ADU-FWHM': use empirical sky in ADU (skyADU), 
                         use empirical FWHM (fwhm)
            Default is 'ADU'.

        Returns
        -------
        SNRout: float
            Signal to noise ratio.
        """
    
        # DECam filters central wavelengths
        cwl_filt_nm = np.asarray([self.u, self.g, self.r, 
                                  self.i, self.z, self.Y])

        # DECam filters CCD_eff and primary_reflectivity
        fct_CCD_eff = interp1d(cwl_filt_nm, self.CCD_eff_filt)
        CCD_eff = fct_CCD_eff(cwl_nm)
        fct_prim_refl = interp1d(cwl_filt_nm, self.prim_refl_filt)
        primary_refl = fct_prim_refl(cwl_nm)
        
        # scaling between seeing at different filters
        seeing_filt_arcsec = np.asarray([self.seeing['u'], self.seeing['g'], 
                                         self.seeing['r'], self.seeing['i'],
                                         self.seeing['z'], self.seeing['Y']])
        fct_seeing = interp1d(cwl_filt_nm, seeing_filt_arcsec)
        seeing_arcsec = fct_seeing(cwl_nm)
    
        # filter width in Hz
        bandpass_Hz = self.cspeed / (cwl_nm - bandpass_nm / 2.) - \
                      self.cspeed / (cwl_nm + bandpass_nm / 2.)  # Hz

        # hc / lambda in MKS (note cspeed and cwl_nm are in nm)
        hc_lambda = self.hPlanck_MKS * self.cspeed / cwl_nm
    
        # magnitude scale zero point flux
        # photons / sec / m2  #  taken from ETC
        zero_photrate = 3.631e-23 * bandpass_Hz / hc_lambda 
    
        # primary mirror effective area
        area = np.pi*(self.diameter/2.0)**2.0 # m2 

        # photons from zero magnitude source
        zero_phot = zero_photrate * area * exptime  # photons
    
        # filter, corrector and atmospheric transmission
        all_atmosph_eff = self.num_atm_eff * np.exp(1.) / np.exp(airmass)
        fct_filt_eff = interp1d(cwl_filt_nm, self.all_filter_eff)
        filter_eff = fct_filt_eff(cwl_nm)
        
        fct_corr_eff = interp1d(cwl_filt_nm, self.all_corrector_eff)
        corrector_eff = fct_corr_eff(cwl_nm)
        
        fct_atm_eff = interp1d(cwl_filt_nm, all_atmosph_eff)
        atmosph_eff = fct_atm_eff(cwl_nm)
        
        # derive FWHM from seeing and airmass or use empirical FWHM
        if skymode == 'mag' or skymode == 'ADU':
            # seeing correction factor due to airmass
            fseeing_airmass = 1. / np.cos(np.arccos(1. / airmass))**(3./5.) 

            # airmass and optics effect on FWHM
            FWHM_arcsec = np.sqrt((seeing_arcsec * fseeing_airmass)**2 + 0.63**2)

        elif skymode == 'mag-FWHM' or skymode == 'ADU-FWHM':
            # if FWHM is provided do not scale seeing
            FWHM_arcsec = fwhm

        else:
            raise ValueError("SNR: skymode %s not recognized" % skymode)
    
        # aperture in arcsec2
        aperture = np.pi * ((self.aperture_eff * FWHM_arcsec) / 2.)**2  
    
        # final throughput
        throughput = np.prod([CCD_eff, filter_eff, corrector_eff,
                              primary_refl, self.vig, atmosph_eff])

        # electrons from a zero mag source
        zero_signal = zero_phot * throughput 

        # electrons from source
        # (SHOULD CALCULATE AT GIVEN WAVELENGTH DEPENDING ON SPECTRUM!)
        # electrons from a source of the given magnitude
        source_electrons = zero_signal * 10**(mag / -2.5) 
    
        # electrons from the sky: 
        # 1) sky_ADU is given (use gain, aperture, pixel scale)
        # 2) sky_mag is given (use zero_signal, aperture and airmass)
        if skymode == 'ADU' or skymode == 'ADU-FWHM':
            # electrons from the sky per aperture given the 
            # empirical sky per pixel in ADU
            sky_electrons = skyADU * self.gain * aperture / self.pixelsize ** 2

        elif skymode == 'mag' or skymode == 'mag-FWHM':
            # electrons from the sky per aperture
            sky_electrons = np.prod([10**(-skymag / 2.5), 
                                     zero_signal, aperture, airmass]) 
        else:
            raise ValueError("SNR: skymode %s not recognized" % skymode)
    
        # readout noise per aperture
        # electrons^2/pixel^2
        RON_aper = self.RON_pix ** 2 * (aperture / self.pixelsize ** 2) 
            
        # surce signal to noise ratio
        SNRout = source_electrons / np.sqrt(source_electrons + \
                                            sky_electrons + nread * RON_aper)
    
        return SNRout
    
    def findmag(self, exptime: float, SNRin: float, cwl_nm=500, 
                bandpass_nm=1.0, band='r', airmass=1.25, skymode='ADU',
                skymag=22.0, nread=1, fwhm=1.0, skyADU=120):
        """Calculate magnitude.

        Parameters
        ----------
        exptime: float
            Exposure time in secs.
        SNRin: float
            Target signal to noise ratio.
        airmass: float (optional)
            Airmass. Default is 1.25.
        band: str (optional)
            Broad band filter. Default is 'r'.
        bandpass_nm: float
            With of bandpass in nm. Default is 1.0.
        cwl_nn: float (optional)
            Central wavelength in nm. Default is 500.
        fwhm: float (optional)
            FWHM. Default is 1.0.
        nread: int (optional)
            Number of times CCD is readout, i.e. for higher readout rates, 
            this increases the noise. Default: 1
        skyADU: float (optional)
            Sky brightness in ADU. Default is 120. 
            Only used if 'skymode' \in ['ADU', 'ADU-FWHM'].
        skymag: float (optional)
            Sky brightness in magnitude. Default is 22.0. 
        skymode: str (optional)
            If mag: sky given in mag per arcsec2 at zenith (skymag)
                    FWHM derived from seeing in r band, band and airmass
            If ADU: use empirical sky in ADU (skyADU), FWHM derived from
                    seeing in r band, band and airmass
            If mag-FWHM: sky given in mag per arcsec2 at zenith (skymag),
                         use empirical FWHM (fwhm)
            If ADU-FWHM: use empirical sky in ADU (skyADU), 
                         use empirical FWHM (fwhm)
            Default is 'ADU'.

        Returns
        -------
        magsarray: float
            Magnitude corresponding to input SNR and exptime.    
        """        

        if skymode == 'mag':
            SNRs = self.SNR(band=band, mag=self.magsarray, exptime=exptime,
                            nread=nread, airmass=airmass, skymode=skymode,
                            skymag=skymag, cwl_nm=cwl_nm,
                            bandpass_nm=bandpass_nm)
        elif skymode == 'mag-FWHM':
            SNRs = self.SNR(band=band, mag=self.magsarray, exptime=exptime, 
                            nread=nread, airmass=airmass, skymode=skymode,
                            skymag=skymag, cwl_nm=cwl_nm, bandpass_nm=bandpass_nm,
                            fwhm=fwhm)
        elif skymode == 'ADU':
            SNRs = self.SNR(band=band, mag=self.magsarray, exptime=exptime, 
                            nread=nread, airmass=airmass, skymode=skymode,
                            skyADU=skyADU)
        elif skymode == 'ADU-FWHM':
            SNRs = self.SNR(band=band, mag=self.magsarray, exptime=exptime, 
                            nread=nread, airmass=airmass, skymode=skymode, 
                            skyADU=skyADU,fwhm=fwhm)
        else:
            raise ValueError("Ivalid skymode value: options are 'mag'," + \
                             "'mag-FWHM', 'ADU' and 'ADU-FWHM'.")
    
        return magsarray[np.argmin((SNRs - SNRin)**2)]

    def findexptime(self, mag: float, SNRin:float, cwl_nm=500,
                    bandpass_nm=1.0, band='r', airmass=1.25, skymode='ADU',
                    skymag=22.0, nread=1, skyADU=120, fwhm=1.0):
        """Find required exposure time.
    
        Parameters
        ----------
        mag: float
            Magnitude.
        SNRin: float
            Target signal to noise ratio.
        airmass: float (optional)
            Airmass. Default is 1.25.
        band: str (optional)
            Broad band filter. Default is 'r'.
        bandpass_nm: float
            With of bandpass in nm. Default is 1.0.
        cwl_nn: float (optional)
            Central wavelength in nm. Default is 500.
        fwhm: float (optional)
            FWHM. Default is 1.0.
        nread: int (optional)
            Number of times CCD is readout, i.e. for higher readout rates, 
            this increases the noise. Default: 1
        skyADU: float (optional)
            Sky brightness in ADU. Default is 120. 
            Only used if 'skymode' \in ['ADU', 'ADU-FWHM'].
        skymag: float (optional)
            Sky brightness in magnitude. Default is 22.0. 
        skymode: str (optional)
            If mag: sky given in mag per arcsec2 at zenith (skymag)
                    FWHM derived from seeing in r band, band and airmass
            If ADU: use empirical sky in ADU (skyADU), FWHM derived from
                    seeing in r band, band and airmass
            If mag-FWHM: sky given in mag per arcsec2 at zenith (skymag),
                         use empirical FWHM (fwhm)
            If ADU-FWHM: use empirical sky in ADU (skyADU), 
                         use empirical FWHM (fwhm)
            Default is 'ADU'.

        Returns
        -------
        est_exptime: float
            Estimated exposure time.
        """  
            
        if skymode == 'mag':
            SNRs = self.SNR(band=band, mag=mag, exptime=self.texparray,
                            nread=nread, airmass=airmass, skymode=skymode, 
                            skymag=skymag, cwl_nm=cwl_nm, 
                            bandpass_nm=bandpass_nm)
        elif skymode == 'mag-FWHM':
            SNRs = self.SNR(band=band, mag=mag, exptime=self.texparray,
                            nread=nread, airmass=airmass, skymode=skymode,
                            skymag=skymag, cwl_nm=cwl_nm,
                            bandpass_nm=bandpass_nm, fwhm=fwhm)
        elif skymode == 'ADU':
            SNRs = self.SNR(band=band, mag=mag, exptime=self.texparray,
                            nread=nread, airmass=airmass, skymode=skymode,
                            skyADU=skyADU)
        elif kwargs['skymode'] == 'ADU-FWHM':
            SNRs = self.SNR(band=band, mag=mag, exptime=self.texparray, 
                            nread=nread, airmass=airmass, skymode=skymode,
                            skyADU=skyADU, fwhm=fwhm)

        else:
            raise ValueError('Invalid "skymode" value. Possible options ' + \
                             'are: "mag", "mag-FWHM", "ADU" and "ADU-FWHM".')

        est_exptime = self.texparray[np.argmin((SNRs - SNRin)**2)]

        return est_exptime


def main():
    return None


if __name__ == '__main__':
    main()

