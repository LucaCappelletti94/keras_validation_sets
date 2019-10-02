keras_validation_sets
=========================================================================================
|travis| |sonar_quality| |sonar_maintainability| |codacy| |code_climate_maintainability| 

Python package offering a callback for handling multiple validation sets.

How do I install this package?
----------------------------------------------
As usual, just download it using pip:

.. code:: shell

    pip install keras_validation_sets

Tests Coverage
----------------------------------------------
Since some software handling coverages sometime get slightly different results, here's three of them:

|coveralls| |sonar_coverage| |code_climate_coverage|

Stuff to do
----------------------------------------------
- [ ] Add all the code from the various versions you have made into a ./implementations directory.
- [ ] Parse through said code and determine a structure for:
    - [ ] Validation sets created though generators
    - [ ] "Normal" validation sets
- [ ] Code and test

.. |travis| image:: https://travis-ci.org/LucaCappelletti94/keras_validation_sets.png
   :target: https://travis-ci.org/LucaCappelletti94/keras_validation_sets
   :alt: Travis CI build

.. |sonar_quality| image:: https://sonarcloud.io/api/project_badges/measure?project=LucaCappelletti94_keras_validation_sets&metric=alert_status
    :target: https://sonarcloud.io/dashboard/index/LucaCappelletti94_keras_validation_sets
    :alt: SonarCloud Quality

.. |sonar_maintainability| image:: https://sonarcloud.io/api/project_badges/measure?project=LucaCappelletti94_keras_validation_sets&metric=sqale_rating
    :target: https://sonarcloud.io/dashboard/index/LucaCappelletti94_keras_validation_sets
    :alt: SonarCloud Maintainability

.. |sonar_coverage| image:: https://sonarcloud.io/api/project_badges/measure?project=LucaCappelletti94_keras_validation_sets&metric=coverage
    :target: https://sonarcloud.io/dashboard/index/LucaCappelletti94_keras_validation_sets
    :alt: SonarCloud Coverage

.. |coveralls| image:: https://coveralls.io/repos/github/LucaCappelletti94/keras_validation_sets/badge.svg?branch=master
    :target: https://coveralls.io/github/LucaCappelletti94/keras_validation_sets?branch=master
    :alt: Coveralls Coverage

.. |codacy|  image:: https://api.codacy.com/project/badge/Grade/b78d67845fe24f81919d95686ffb5bf8
    :target: https://www.codacy.com/manual/LucaCappelletti94/keras_validation_sets?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=LucaCappelletti94/keras_validation_sets&amp;utm_campaign=Badge_Grade
    :alt: Codacy Maintainability

.. |code_climate_maintainability| image:: https://api.codeclimate.com/v1/badges/45a6f2d0b8a7b2909974/maintainability
    :target: https://codeclimate.com/github/LucaCappelletti94/keras_validation_sets/maintainability
    :alt: Maintainability

.. |code_climate_coverage| image:: https://api.codeclimate.com/v1/badges/45a6f2d0b8a7b2909974/test_coverage
    :target: https://codeclimate.com/github/LucaCappelletti94/keras_validation_sets/test_coverage
    :alt: Code Climate Coverate
