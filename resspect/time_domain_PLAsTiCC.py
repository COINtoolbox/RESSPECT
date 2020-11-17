# Copyright 2020 resspect software
# Author: Emille E. O. Ishida
#     
# created on 26 February 2020
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

import os

from copy import deepcopy
import numpy as np
import pandas as pd

from resspect import LightCurve


class PLAsTiCCPhotometry(object):
    """Handles photometric information for the PLAsTiCC data.

    Attributes
    ----------
    bazin_header: str
        Header to be added to features files for each day.
    class_code: dict
        Keywords are model numbers, values are model names. 
    fdic: dict
        Keywords: ['test', 'train'], values are names of photo files.
    include_classes: list
        List of classes to be considered.
    max_epoch: float
        Maximum MJD for the entire data set.
    metadata: pd.DataFrame
        Metadata from PLAsTiCC zenodo test sample.
    min_epoch: float
        Minimum MJD for the entire data set.
    rmag_lim: float
        Maximum r-band magnitude allowing a query. 
    
    Methods
    -------
    build_one_epoch(raw_data_dir: str, day_of_survey: int,
                    time_domain_dir: str, feature_method: str)
        Construct 1 entire day of observation.
    create_all_daily_files(raw_data_dir: str)
        Create 1 file per day for all days of the survey. 
    create_daily_file(output_dir: str, day: int, vol: int, header: str)
        Create one file for a given day of the survey. Contains only header.    
    fit_one_lc(raw_data_dir: str, snid: int, sample: str)
        Fit one light curve throughout the entire survey.
    read_metadata(path_to_data_dir: str, classes: list)
        Read metadata and filter only required classes.
    write_bazin_to_file(lightcurve: LightCurve, features_file: str)
        Write Bazin parameters and metadata to file.
    """

    def __init__(self):
        self.class_code = {90: 'Ia', 67: '91bg', 52:'Iax', 42:'II', 62:'Ibc', 
             95: 'SLSN', 15:'TDE', 64:'KN', 88:'AGN', 92:'RRL', 65:'M-dwarf',
             16:'EB',53:'Mira', 6:'MicroL', 991:'MicroLB', 992:'ILOT', 
             993:'CART', 994:'PISN',995:'MLString'}
        self.fdic = {}
        self.fdic['test'] = ['plasticc_test_lightcurves_' + str(x).zfill(2) + '.csv.gz' 
                              for x in range(1, 12)]
        self.fdic['train'] = ['plasticc_train_lightcurves.csv.gz']
        self.include_classes = [] 
        self.max_epoch = 60675
        self.metadata = pd.DataFrame()
        self.min_epoch = 59580
        self.rmag_lim = 24
        
    def create_daily_file(self, output_dir: str,
                          day: int, header='Bazin',
                          get_cost=False):
        """Create one file for a given day of the survey.

        The file contains only header for the features file.

        Parameters
        ----------
        output_dir: str
            Complete path to output directory.
        day: int
            Day passed since the beginning of the survey.
        header: str (optional)
            List of elements to be added to the header.
            Separate by 1 space.
            Default option uses header for Bazin features file.
        get_cost: bool (optional)
            If True, calculate cost of taking a spectra in the last 
            observed photometric point. Default is False.
        """

        features_file = output_dir + 'day_' + str(day) + '.dat'

        if header == 'Bazin':
            if get_cost:
                self.bazin_header = 'id redshift type code orig_sample queryable ' + \
                            'last_rmag cost_4m cost_8m gA gB gt0 ' + \
                            'gtfall gtrise rA rB rt0 rtfall rtrise iA ' + \
                            'iB it0 itfall itrise zA zB zt0 ztfall ztrise\n'
            else:
                self.bazin_header = 'id redshift type code sample queryable last_rmag uA uB ' + \
                               'ut0 utfall utrise gA gB gt0 gtfall ' + \
                               'gtrise rA rB rt0 rtfall rtrise iA iB it0 ' + \
                               'itfall itrise zA zB zt0 ztfall ztriseYA ' + \
                               'YB Yt0 Ytfall Ytrise\n'
            
            # add headers to files
            with open(features_file, 'w') as param_file:
                param_file.write(self.bazin_header)

        else:
            raise ValueError('Only Bazin headers are implemented.')


    def read_metadata(self, path_to_data_dir: str, classes: list,
                      field='DDF'):
        """ Read metadata and filter only required classes.

        Populates the metadata attribute. 

        Parameters
        ----------
        path_to_data_dir: str
            Directory containing all PlAsTiCC zenodo files. 
        classes: list of int
            Codes for classes we wish to keep. 
        field: str (optional)
            Telescope cadence.
            'DDF', 'WFD' or 'DDF+WFD'. Default is 'DDF'.
        """
        fname = 'plasticc_test_metadata.csv.gz'

        # store classes information
        self.include_classes = classes

        # read header information
        header = pd.read_csv(path_to_data_dir + fname, 
                             index_col=False)
        if ' ' in header.keys()[0]:
            header = pd.read_csv(path_to_data_dir + fname, 
                                 sep=' ', index_col=False)
            
        # remove useless columns
        header2 = header[['object_id', 'true_target', 'true_z',
                         'true_peakmjd', 'true_distmod']]

        class_flag = np.array([header2['true_target'].iloc[j] in classes
                               for j in range(header.shape[0])])

        if field in ['WFD', 'DDF']:
            ddf_flag = np.array([bool(item) 
                                 for item in header['ddf_bool'].values])
            
            if field == 'DDF':
                final_flag = np.logical_and(ddf_flag, class_flag)
                self.metadata = header2[final_flag]
            elif field == 'WFD':
                final_flag2 = np.logical_and(~ddf_flag, class_flag)
                self.metadata = header2[final_flag2]
        else:
            self.metadata = header2[class_flag]

    def create_all_daily_files(self, output_dir:str,
                               get_cost=False):
        """Create 1 file per day for all days of the survey. 
        
        Each file contains only the header. 

        Parameters
        ----------
        output_dir: str
            Output directory.
        get_cost: bool (optional)
            If True, calculate cost of taking a spectra in the last 
            observed photometric point. Default is False.
        """
        
        ## create daily files
        # run through all days of the survey
        for day_of_survey in range(1, self.max_epoch - self.min_epoch):
            self.create_daily_file(output_dir=output_dir,
                                   day=day_of_survey, get_cost=get_cost)

    def write_bazin_to_file(self, lc: LightCurve, 
                            features_file: str, tel_names=['4m', '8m'], 
                            get_cost=False):
        """Write Bazin parameters and metadata to file.

        Use output filename defined in the features_file attribute.

        Parameters
        ----------
        lc: LightCurve
            resspect light curve object.
        features_file: str
            Output file to store Bazin features.
        queryable: bool
            Rather this object is available for querying.
        tel_names: list (optional)
            Names of the telescopes under consideraton for spectroscopy.
            Only used if "get_cost == True".
            Default is ["4m", "8m"].
        get_cost: bool (optional)
            If True, calculate cost of taking a spectra in the last 
            observed photometric point. Default is False.
        
        Returns
        -------
        line: str
            A line concatenating metadata and Bazin fits for 1 obj.
        """

        # build an entire line with bazin features
        line = str(lc.id) + ' ' + str(lc.redshift) + ' ' + \
               str(lc.sntype) + ' ' + str(lc.sncode) + \
               ' ' + str(lc.sample) + ' ' + str(lc.queryable) + ' ' + \
               str(lc.last_mag) + ' '
                                   
        if get_cost:
            for k in range(len(tel_names)):
                line = line + str(lc.exp_time[tel_names[k]]) + ' '

        for item in lc.bazin_features:
            line = line + str(item) + ' '

        line = line + '\n'
                                                               
        # save features to file
        with open(features_file, 'a') as param_file:
            param_file.write(line)

        return line
            
    def fit_one_lc(self, raw_data_dir: str, snid: int, 
                   output_dir: str, vol=None, day=None, screen=False,
                   queryable_criteria=1, days_since_last_obs=2,
                   get_cost=False, tel_sizes=[4, 8], tel_names=['4m', '8m'],
                   feature_method='Bazin', spec_SNR=10, **kwargs):
        """Fit one light curve throughout the entire survey.

        Save results to appropriate file, considering 1 day survey 
        evolution. 

        Parameters
        ----------
        raw_data_dir: str
            Complete path to all PLAsTiCC zenodo files.
        snid: int
            Object id for the transient to be fitted.
        output_dir:
            Directory to store output time domain files.
        vol: int or None (optional)
            Index of the original PLAsTiCC zenodo light curve
            files where the photometry for this object is stored.
            If None, search for id in all light curve files.
            If training sample, choose "vol = 0".
            Default is None.
        day: int or None (optional)
            Day since beginning of survey to be considered.
            If None, fit all days. Default is None.
        screen: bool (optional)
            Print steps evolution on screen.
            Default is False.
        queryable_criteria: int [1, 2 or 3] (optional)
            Criteria to determine if an obj is queryable.
            1 -> Cut on last measured photometric point.
            2 -> if last obs was further than days_since_last_obs, 
                 use Bazin estimate for today. Otherwise, use
                 the last observed point.
            Default is 1.
        days_since_last_obs: int (optional)
            If there is an observation within these days, use the
            measured value, otherwise estimate current mag.
            Only used if "criteria == 2". Default is 2.
        get_cost: bool (optional)
            If True, calculate cost of taking a spectra in the last 
            observed photometric point. Default is False.
        tel_names: list (optional)
            Names of the telescopes under consideraton for spectroscopy.
            Only used if "get_cost == True".
            Default is ["4m", "8m"].
        tel_sizes: list (optional)
            Primary mirrors diameters of potential spectroscopic telescopes.
            Only used if "get_cost == True".
            Default is [4, 8].
        feature_method: str (optional)
            Feature extraction method.
            Only possibility is 'Bazin'.
        spec_SNR: float (optional)
            SNR required for spectroscopic follow-up. Default is 10.
        kwargs: extra parameters
            Any input required by ExpTimeCalc.findexptime function.
        """
        
        # check if telescope names are ok
        if ('cost_' + tel_names[0] not in self.bazin_header or \
            'cost_' + tel_names[1] not in self.bazin_header) and get_cost: 
                raise ValueError('Telescope names are hard coded in ' + \
                                 'header.\n Change attribute ' + \
                                 '"bazin_header" to continue!')

        # store number of points per day
        npoints = {}
        npoints[0] = 0         

        if vol ==  None:
            vol = 0
            cont = True
            
            # search within test light curve files
            while cont:
                vol = vol + 1
                if screen:
                    print('vol: ', vol)
                
                # create light curve instance
                orig_lc = LightCurve() 
                orig_lc.load_plasticc_lc(raw_data_dir + self.fdic['test'][vol - 1], snid)
       
                if vol == 10 and orig_lc.photometry['mjd'].values.shape[0] == 0:
                    raise ValueError('Light curve for ', snid, ' not found!')
                
                elif orig_lc.photometry['mjd'].values.shape[0] > 0:
                    cont = False

        elif isinstance(vol, int):
            if vol == 0:
                # load light curve
                orig_lc.load_plasticc_lc(raw_data_dir + self.fdic['train'][vol], snid)
                
            else:
                # load light curve
                orig_lc.load_plasticc_lc(raw_data_dir + self.fdic[vol - 1], snid)
            
                if len(orig_lc.photometry['mjd'].values.shape[0]) == 0:
                    raise ValueError('Light curve for ', snid, ' not found!')
            

        # define days in which this light curve exists
        min_mjd = min(orig_lc.photometry['mjd'].values)
        max_mjd = max(orig_lc.photometry['mjd'].values)

        # get number of points in all days of the survey for this light curve
        for days in range(1, self.max_epoch - self.min_epoch):

            # see which epochs are observed until this day
            now = days + self.min_epoch
            photo_flag_now = orig_lc.photometry['mjd'].values <= now

            # number of points today
            npoints[days] = sum(photo_flag_now)

        if day == None:
            # for every day of survey
            fit_days = range(int(min_mjd - self.min_epoch), 
                             int(max_mjd - self.min_epoch) + 1)
            
        elif isinstance(day, int):
            # for a specific day of the survey
            fit_days = [day]
            
        # create instance to store Bazin fit parameters    
        line = False
            
        for day_of_survey in fit_days:

            lc = deepcopy(orig_lc)

            # see which epochs are observed until this day
            today = day_of_survey + self.min_epoch
            photo_flag = lc.photometry['mjd'].values <= today

            # number of points today
            npoints[day_of_survey] = sum(photo_flag)                                 

            # check if any point survived, other checks are made
            # inside lc object
            if npoints[day_of_survey] > 4:
                
                # only the allowed photometry
                lc.photometry = lc.photometry[photo_flag] 
            
                # perform feature extraction
                if feature_method == 'Bazin':
                    lc.fit_bazin_all()
                else:
                    raise ValueError('Only Bazin features are implemented!')
                    
                # only save to file if all filters were fitted
                if len(lc.bazin_features) > 0 and \
                        'None' not in lc.bazin_features:
                    
                    if screen:
                        print('Processing day: ', day_of_survey)
                        
                    # calculate r-mag today
                    lc.queryable = lc.check_queryable(mjd=self.min_epoch + day_of_survey,
                                       filter_lim=self.rmag_lim, 
                                       criteria=queryable_criteria,
                                       days_since_last_obs=days_since_last_obs)
                    

                    if get_cost:
                        for k in range(len(tel_names)):
                            lc.calc_exp_time(telescope_diam=tel_sizes[k],
                                             telescope_name=tel_names[k],
                                             SNR=spec_SNR, **kwargs)
                            
                        # see if query is possible
                        query_flags = []
                        for item in tel_names:
                            if lc.exp_time[item] < 7200:
                                query_flags.append(True)
                            else:
                                query_flags.append(False)
                            
                        lc.queryable = bool(sum(query_flags))

                    if lc.queryable:
                        # mask only metadata for this object
                        mask = self.metadata['object_id'].values == snid
                    
                        # set redshift
                        lc.redshift = self.metadata['true_z'].values[mask][0]

                        # set light curve type
                        lc.sncode  = self.metadata['true_target'].values[mask][0]
                        lc.sntype = self.class_code[lc.sncode]

                        # set id
                        lc.id = snid

                        # set sample
                        lc.sample = 'pool'

                        # set filename
                        features_file = output_dir + 'day_' + \
                                             str(day_of_survey) + '.dat'

                        # write to file
                        line = self.write_bazin_to_file(lc, features_file, 
                                                        tel_names=tel_names,
                                                        get_cost=get_cost)

                        if screen:
                            print('   *** Wrote to file ***   ')
