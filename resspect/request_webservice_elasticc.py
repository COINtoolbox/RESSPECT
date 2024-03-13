# Copyright 2020 resspect software
# Author: Amanda Wasserman
#
# created on 12 March 2024
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

import requests

class TomClient:
        def __init__(self, url='https://desc-tom.lbl.gov', username=None, 
                     password=None, passwordfile = None, connect=True):
            self._url = url
            self._username = username
            self._password = password
            self._rqs=None
            if self._password is None:
                 if passwordfile is None:
                      raise RuntimeError('No password or passwordfile provided')
                 with open(passwordfile) as ifp:
                      self._password = ifp.readline().strip()

            if connect:
                self.connect()

        def connect(self):
            self._rqs = requests.session()  #should this be capitalized S?
            res = self._rqs.get(f'{self._url}/acounts/login/')
            if res.status_code != 200:
                raise RuntimeError(f'Failed to connect to {self._url}')
            res = self._rqs.post(f'{self._url}/accounts/login/', 
                                 data={'username':self._username, 
                                       'password':self._password,
                                       'csrfmiddlewaretoken': self._rqs.cookies['csrftoken']})
            if res.status_code != 200:
                 raise RuntimeError(f'Failed to login.')
            if 'Please enter a correct' in res.text:
                 raise RuntimeError("failed to log in.")
            self._rqs.headers.update({'X-CSRFToken': self._rqs.cookies['csrftoken']})

        def request(self, method="GET", page=None, **kwargs):
             
             return self._rqs.request(method=method, url=f"{self._url}/{page}", **kwargs)

#def request(website: str):
#    r = requests.get(website)
#    status = r.status_code
#    text = r.text
#    return text