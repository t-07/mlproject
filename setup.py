#This file is used to create your ML project as a package. So you can install it as package and also we 
# can deploy it on Python Pi Pi and anybody will be able to install the project as package just like we 
# do other packages like seaborn, matplotlib.


from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->List[str]:

    '''This function returns the list of requirements'''
    HYPHEN_E_DOT='-e .'
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements




setup(
    name="mlproject",
    version="0.0.1",
    packages=find_packages(), #searches the folders that has __init__.py file to see which packages are needed
    #install_requires=['numpy','pandas','seaborn'] # this is not feasible if we have 100 packages
    install_requires=get_requirements('requirements.txt')
)