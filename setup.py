from setuptools import setup, find_packages
setup(name='suricate',
      version='0.63',
      description='MAAAAAAAAAAH',
      url='http://github.com/ogierpaul/suricate',
      author='Flying Circus',
      author_email='flyingcircus@example.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'scikit-learn',
          'pandas',
          'numpy',
          'fuzzywuzzy',
          'elasticsearch', 'bleach', 'py2neo', 'sqlalchemy'
      ],
      package_data={'suricate': ['data/companydata/*.csv'],
                    'pacoetl': ['utils/countrydata/*.csv']},
      zip_safe=False
)

