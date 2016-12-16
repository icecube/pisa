# Author: Philipp Eller
# Email:  pde3@psu.edu
"""
Configuration parser class that subclasses ConfigParser.SafeConfigParser to
handle PISA's specific config-file-parsing needs...

"""

import ConfigParser
import re

from pisa.utils import resources


__all__ = ['BetterConfigParser']


class BetterConfigParser(ConfigParser.SafeConfigParser):

    def read(self, fname):
        # preprocessing for include statements
        new_fnames = [fname]
        processed_fnames = []
        # loop until we cannot find any more includes
        while True:
            processed_fnames.extend(new_fnames)
            new_fnames = self.recursive_fnames(new_fnames)
            rec_incs = set(new_fnames).intersection(processed_fnames)
            if any(rec_incs):
                raise ValueError('Recursive include statements found for %s'
                                 % ', '.join(rec_incs))
            if len(new_fnames) == 0:
                break
        # call read with complete files list
        ConfigParser.SafeConfigParser.read(self, processed_fnames)

    def recursive_fnames(self, fnames):
        new_fnames = []
        for fname in fnames:
            new_fnames.extend(self.process_fnames(fname))
        return new_fnames

    def process_fnames(self, fname):
        processed_fnames = []
        with open(fname) as f:
            for line in f.readlines():
                if line.startswith('#include '):
                    inc_file = line[9:].rstrip()
                    inc_file = resources.find_resource(inc_file)
                    processed_fnames.append(inc_file)
                else:
                    break
        return processed_fnames

    def get(self, section, option):
        result = ConfigParser.SafeConfigParser.get(self, section, option,
                                                   raw=True)
        result = self.__replaceSectionwideTemplates(result)
        return result

    def items(self, section):
        config_list = ConfigParser.SafeConfigParser.items(
            self, section=section, raw=True
        )
        result = [(key, self.__replaceSectionwideTemplates(value)) for
                  key, value in config_list]
        return result

    def optionxform(self, optionstr):
        """Enable case sensitive options in .ini/.cfg files."""
        return optionstr

    def __replaceSectionwideTemplates(self, data):
        """Replace <section|option> with get(section, option) recursivly."""
        result = data
        findExpression = re.compile(r"((.*)\<!(.*)\|(.*)\!>(.*))*")
        groups = findExpression.search(data).groups()

        # If expression not matched
        if groups != (None, None, None, None, None):
            result = self.__replaceSectionwideTemplates(groups[1])
            result += self.get(groups[2], groups[3])
            result += self.__replaceSectionwideTemplates(groups[4])
        return result
