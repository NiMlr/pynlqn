from setuptools import setup

setup(name='pynlqn',
      version='0.1',
      description='A non-local quasi-newton method for global optimization.',
      url='http://github.com/NiMlr/pynlqn',
      author='Nils Müller',
      license='MIT',
      packages=['pynlqn'],
      install_requires = ['numpy', 'scipy'],
      zip_safe=False)
