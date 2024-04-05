# Copyright 2020 resspect software
# Author: Rob Knop
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
    """A thin class that supports sending requests via "requests" to the DESC tom.

    Usage: initialize one of these, giving it the url, your TOM
    username, and either your TOM password, or a file that has your TOM
    password in it:

      tc = TomClient( username='rknop', passwordfile='/home/raknop/secrets/tom_rknop_passwd' )

    (You can give it a url with url=; it defaults to https://desc-tom.lbl.gov.)

    Thereafter, just do something like

      res = tc.request( "POST", "elasticc2/ppdbdiaobject/55772173" )

    and res will come back with a string that you can load into JSON
    that will have all the fun data about PPDBDiaObject number 55772173.

    tc.request is just a thin front-end to python requests.request.  The
    only reason to use this client rather than the python requests
    module directly is that this class takes care of the stupid fiddly
    bits of getting some headers that django demands set up right in the
    request object when you log in.

    """

    def __init__( self, url="https://desc-tom.lbl.gov", username=None, password=None, passwordfile=None, connect=True ):
        self._url = url
        self._username = username
        self._password = password
        self._rqs = None
        if self._password is None:
            if passwordfile is None:
                raise RuntimeError( "Must give either password or passwordfile. " )
            with open( passwordfile ) as ifp:
                self._password = ifp.readline().strip()

        if connect:
            self.connect()

    def connect( self ):
        self._rqs = requests.session()
        res = self._rqs.get( f'{self._url}/accounts/login/' )
        if res.status_code != 200:
            raise RuntimeError( f"Got status {res.status_code} from first attempt to connect to {self._url}" )
        res = self._rqs.post( f'{self._url}/accounts/login/',
                              data={ 'username': self._username,
                                     'password': self._password,
                                     'csrfmiddlewaretoken': self._rqs.cookies['csrftoken'] } )
        if res.status_code != 200:
            raise RuntimeError( f"Failed to log in; http status: {res.status_code}" )
        if 'Please enter a correct' in res.text:
            # This is a very cheesy attempt at checking if the login failed.
            # I haven't found clean documentation on how to log into a django site
            # from an app like this using standard authentication stuff.  So, for
            # now, I'm counting on the HTML that happened to come back when
            # I ran it with a failed login one time.  One of these days I'll actually
            # figure out how Django auth works and make a version of /accounts/login/
            # designed for use in API scripts like this one, rather than desgined
            # for interactive users.
            raise RuntimeError( "Failed to log in.  I think.  Put in a debug break and look at res.text" )
        self._rqs.headers.update( { 'X-CSRFToken': self._rqs.cookies['csrftoken'] } )

    def request( self, method="GET", page=None, **kwargs ):
        """Send a request to the TOM

        method : a string with the HTTP method ("GET", "POST", etc.)

        page : the page to get; this is the URL but with the url you
          passed to the constructor removed.  So, if you wanted to get
          https://desc-tom.lbl.gov/elasticc, you'd pass just "elasticc"
          here.

        **kwargs : additional keyword arguments are passed on to
          requests.request

        """
        return self._rqs.request( method=method, url=f"{self._url}/{page}", **kwargs )
        
    def post( self, page=None, **kwargs ):
        """Shortand for TomClient.request( "POST", ... )"""
        return self.request( "POST", page, **kwargs )

    def get( self, page=None, **kwargs ):
        """Shortand for TomClient.request( "GET", ... )"""
        return self.request( "GET", page, **kwargs )

    def put( self, page=None, **kwargs ):
        """Shortand for TomClient.request( "PUT", ... )"""
        return self.request( "PUT", page, **kwargs )

