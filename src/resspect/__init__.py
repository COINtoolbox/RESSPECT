# Copyright 2020 resspect software
# Author: Emille E. O. Ishida
# created on 14 April 2020
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

from .bazin import *
from .build_plasticc_canonical import *
from .build_plasticc_metadata import *
from .build_snpcc_canonical import *
from .classifiers import *
from .cosmo_metric_utils import *
from .database import *
from .exposure_time_calculator import *
from .fit_lightcurves import *
from .learn_loop import *
from .metrics import *
from .plot_results import *
from .query_strategies import *
from .samples_utils import *
from .snana_fits_to_pd import *
from .scripts.build_canonical import build_canonical as build_canonical
from .scripts.build_time_domain_snpcc import build_time_domain_snpcc as build_time_domain_snpcc
from .scripts.build_time_domain_plasticc import build_time_domain_plasticc as build_time_domain_plasticc
from .scripts.calculate_cosmology_metric import calculate_cosmology_metric as calculate_cosmology_metric
from .scripts.fit_dataset import fit_dataset as fit_dataset
from .scripts.make_metrics_plots import make_metrics_plots as make_metrics_plots
from .scripts.run_loop import run_loop as run_loop
from .scripts.run_time_domain import run_time_domain as run_time_domain
from .time_domain_plasticc import *
from .time_domain_snpcc import *
from .time_domain_loop import *
from .batch_functions import *
from .query_budget_strategies import *
from .bump import *
from .feature_extractors.malanchev import *

import importlib.metadata

__version__ = importlib.metadata.version("resspect")


__all__ = ['accuracy',
           'assign_cosmo',
           'bazin',
           'build_canonical',
           'build_plasticc_canonical',
           'build_plasticc_metadata',
           'build_snpcc_canonical',
           'bump',
           'calculate_SNR',
           'Canonical',
           'CanonicalPLAsTiCC',
           'Canvas',
           'column_deriv_m',
           'compare_two_fishers',
           'cosmo_metric',
           'DataBase',
           'ExpTimeCalc',
           'efficiency',
           'errfunc',
           'fish_deriv_m',
           'fisher_results',
           'find_most_useful',
           'fit_dataset',
           'fit_bump',
           'fit_scipy',
           'fit_snpcc',
           'fit_plasticc',
           'fom',
           'get_cosmo_metric',
           'get_snpcc_metric',
           'get_SNR_headers',
           'knn',
           'learn_loop',
           'load_dataset',
           'LightCurve',
           'mlp',
           'nbg',
           'PLAsTiCCPhotometry',
           'make_metrics_plots',
           'plot_snpcc_train_canonical',
           'protected_exponent',
           'protected_sig',
           'purity',
           'random_forest',
           'random_sampling',
           'read_features_fullLC_samples',
           'read_fits',
           'run_loop',
           'run_time_domain',
           'sep_samples',
           'SNPCCPhotometry',
           'svm',
           'time_domain_loop',
           'uncertainty_sampling',
           'update_matrix']