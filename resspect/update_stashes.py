# Copyright 2020 resspect software
# Author: Amanda Wasserman
#
# created on 18 March 2024
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

__all__ = ["update_pool_stash", "update_training_stash"]

### Need to remove old objs at some point (once they are no longer hot)
# Need to remove objects that have been moved from pool to training
def update_pool_stash(current_stash_path: str, new_night_path: str):
    #should we store old features somewhere? makes it easier to add training objs
    #would want to add current MJD, maybe first MJD, and peak MJD

    #read in current stash as list of strings
    with open(current_stash_path, 'r') as f:
        current_stash = f.readlines()
        
    #read in new night as list of strings
    with open(new_night_path, 'r') as f:
        new_night = f.readlines()

    #if id is already in stash, replace it. else append it
    for obj in new_night:
        replaced = 0
        id = obj.split(',')[0]
        for old_obj in current_stash:
            if id in old_obj:
                current_stash[current_stash.index(old_obj)] = obj
                replaced = 1
                break
        if replaced == 0:
            current_stash.append(obj)
    
    #write new stash
    with open(current_stash_path, 'w') as f:
        for obj in current_stash:
            f.write(obj)

def update_training_stash_with_new_classification(current_training_stash_path: str, new_obj_ids: list, 
                          new_obj_classes: list, new_obj_redshifts: list, current_stash_path: str):
    #add new obj id and class for each point on the training obj light curve going forward
    #(how do we want to do this? add A,B,C, etc to the end of the id?)
    with open(current_stash_path, 'r') as f:
        current_stash = f.readlines()
    
    with open(current_training_stash_path, 'r') as f:
        training_stash = f.readlines()

    #find obj in current stash
    for idx, obj in new_obj_ids:
        for old_obj in current_stash:
            if idx in old_obj:
                line = old_obj.split(',')
                break
        line[1] = new_obj_redshifts[idx]
        line[2] = new_obj_classes[idx]
        train_to_append = ','.join(line)
        training_stash.append(train_to_append)

    #write new training stash
    with open(current_training_stash_path, 'w') as f:
        for obj in training_stash:
            f.write(obj)


def update_training_stash_with_new_features(new_night_path: str, current_training_stash_path: str):
    #add new obj id and features for each point on the training obj light curve going forward
    #(how do we want to do this? add A,B,C, etc to the end of the id?)
    with open(new_night_path, 'r') as f:
        new_night = f.readlines()    
    
    with open(current_training_stash_path, 'r') as f:
        training_stash = f.readlines()

    #append training set with new features
    for obj in new_night:
        id = obj.split(',')[0]
        for old_obj in training_stash:
            if id in old_obj:
                #append new features
                old_split = old_obj.split(',')
                split = [id+mjd, old_split[1], old_split[2], 'N/A', 'N/A'] ###we need mjd (or something) to differentiate the ids
                split.extend(obj.split(',')[5:])
                training_stash.append(','.join(split))
    
    #write new stash
    with open(current_training_stash_path, 'w') as f:
        for obj in training_stash:
            f.write(obj)

def remove_training_from_pool(current_stash_path: str, new_obj_ids: list):
    #remove objects that have been moved from pool to training
    with open(current_stash_path, 'r') as f:
        current_stash = f.readlines()
    
    for id in new_obj_ids:
        for obj in current_stash:
            if id in obj:
                current_stash.remove(obj)
    
    #write new stash
    with open(current_stash_path, 'w') as f:
        for obj in current_stash:
            f.write(obj)

def remove_not_hot_objects(current_stash_path: str):
    #remove objects that are no longer hot
    #need MJDs
    #make sure we're not getting old objects from rob or we'll be wasting computing power adding/removing
    with open(current_stash_path, 'r') as f:
        current_stash = f.readlines()

    for obj in current_stash:
        if obj[current_mjd-discovery_mjd] >5 0:
            current_stash.remove(obj)

    #write new stash
    with open(current_stash_path, 'w') as f:
        for obj in current_stash:
            f.write(obj)
    
