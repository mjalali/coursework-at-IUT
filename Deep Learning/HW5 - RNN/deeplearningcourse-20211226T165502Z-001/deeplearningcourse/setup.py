from pkg_resources import DistributionNotFound, get_distribution
from distutils.core import setup


def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None

install_deps = [
    'numpy',
    'regex',
    'tqdm',
    'gym'

]
tf_ver = '2.0.0a'
if get_dist('tensorflow>='+tf_ver) is None and get_dist('tensorflow_gpu>='+tf_ver) is None:
    install_deps.append('tensorflow>='+tf_ver)

setup(
  name = 'deeplearningcourse',         # How you named your package folder (MyLib)
  packages = ['deeplearningcourse'],   # Chose the same as "name"
  keywords = ['deep learning', 'neural networks', 'tensorflow'],   # Keywords that define your package best
  install_requires=install_deps,
  package_data={
      'deeplearningcourse': ['bin/*', 'data/*', 'data/faces/DF/*', 'data/faces/DM/*', 'data/faces/LF/*', 'data/faces/LM/*'],
   },

)
