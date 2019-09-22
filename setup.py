from setuptools import setup

setup(name='kesmarag-single-planar-fault',
      version='0.1.3',
      author='Costas Smaragdakis',
      author_email='kesmarag@gmail.com',
      url='https://github.com/kesmarag/single-planar-fault',
      packages=['kesmarag.spf'],
      package_dir={'kesmarag.spf': './'},
      install_requires=['matplotlib>=3.1.1',
                        'tornado>=6.0.3',
                        'pyproj>=2.3.1',
                        'numpy>=1.12.1'], )
