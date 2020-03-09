# coding=utf-8
# Copyright 2018-2020 EVA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from src.server.server import start_server

from src.utils.logging_manager import Logger
from src.utils.logging_manager import LoggingLevel

class ServerTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_server(self):

        try:
            start_server(host="localhost", port=5432)

        except Exception as e:
            Logger().log(e, LoggingLevel.CRITICAL)


if __name__ == '__main__':
    unittest.main()