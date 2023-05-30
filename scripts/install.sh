
# Install All Constituent Packages

set -eo 

pip install -e "Shared[dev]"
pip install -e "Auto[dev]"
pip install -e "Ckt[dev]"
