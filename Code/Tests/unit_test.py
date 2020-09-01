import unittest
from test_functions import *
import warnings


class Test_Model(unittest.TestCase):

    def test_deployment(self):
        self.assertTrue(checkdeployment())


# the following is not required if call by pytest instead of python
if __name__ == '__main__':
    unittest.main()
