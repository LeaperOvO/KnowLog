from .CosineSimilarityLoss import *
from .SoftmaxLoss import *
from .SingleSoftmaxLoss import *
from .MultipleNegativesRankingLoss import *
from .LogMultipleNegativesRankingLoss import *
from .LogNLMultipleNegativesRankingLoss import *
from .LogNLMultipleNegativesRankingLossOther import *
from .MultipleNegativesSymmetricRankingLoss import *
from .TripletLoss import *
from .MarginMSELoss import MarginMSELoss
from .MSELoss import *
from .ContrastiveLoss import *
from .ContrastiveTensionLoss import *
from .OnlineContrastiveLoss import *
from .MegaBatchMarginLoss import *
from .DenoisingAutoEncoderLoss import *
from .MultilabelSoftMarginLoss import *
from .TokenClassificationLoss import *
from .GleuTokenClassificationLoss import *

# Triplet losses
from .BatchHardTripletLoss import *
from .BatchHardSoftMarginTripletLoss import *
from .BatchSemiHardTripletLoss import *
from .BatchAllTripletLoss import *

# KD losses
from .KDmlmLoss import *
from .KDstuLoss import *

from .MLMLoss import *

# Enhance losses
from .EnhanceLoss import *
from .ChatgptEnhanceLoss import *
from .ChatgptEnhanceLossab import *