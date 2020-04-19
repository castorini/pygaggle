
# We use __init__ to flatten the hierarchy so that we can do:
# > from pygaggle.lib import IdentityReranker
#
# which is less verbose than
# > from pygaggle.lib.IdentityReranker import IdentityReranker
#

from pygaggle.lib.IdentityReranker import IdentityReranker
