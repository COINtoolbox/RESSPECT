# Copyright 2020 resspect software
# Author: Amanda Wasserman
#
# created on 11 March 2024
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

__all__ = ["get_id_type"]

def get_id_type(file: str):
    #read text file
    data = open(file, "r")

    #retrieve lines with id and type
    sns =[]
    for line in data:
        if '.txt' in line:
            sns.append(line)
    
    ids = [id[0:7] for id in sns]

    type = []
    #parse file for type
    for obj in sns:
        indxb = obj.find("('")
        indxe = obj.find(")  ")
        temp = obj[indxb+2:indxe]
        temp = temp.split(',')[0]
        type.append(temp[0:-1])
    return ids, type

