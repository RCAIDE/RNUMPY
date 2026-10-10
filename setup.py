# setup.py
# (c) Copyright 2024 Aerospace Research Community LLC
# 
# Created:  Aug 2024, E. Botero
# Modified: 
#           
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



 # imports    
from setuptools import setup, find_packages

# ----------------------------------------------------------------------------------------------------------------------
#  Write Version
# ----------------------------------------------------------------------------------------------------------------------
def write_version_py(version,filename='RNUMPY/version.py'):
    cnt = """
# THIS FILE IS GENERATED
version = '%(version)s'

"""

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': version})
    finally:
        a.close()        
        

# ----------------------------------------------------------------------------------------------------------------------
#  Install
# ----------------------------------------------------------------------------------------------------------------------

version     = '0.2.0'
date        = 'Dec 10th, 2024'

write_version_py(version)

with open('README.md', encoding='utf-8') as f:
  long_description = f.read()


# run the setup!!!
setup(
    name = 'RNUMPY',
    version = version, 
    description = 'Research Numpy: run either JAX or standard Numpy from a common interface',
    packages = find_packages(),
    include_package_data = True,
    zip_safe  = False,
    install_requires  = ['numpy','scipy'],
    long_description = long_description,
)  