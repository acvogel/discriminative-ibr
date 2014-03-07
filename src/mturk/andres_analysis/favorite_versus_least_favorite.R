



library(reshape)
library(ggplot2)

forced_choice_no_fam_6random_count_ALLS.csv
forced_choice_no_fam_6random_NOcount_ALNC.csv
forced_choice_no_fam_4random_count_12_january_least_LEAS.csv


## With "Favorite"
favorite <- read.csv("forced_choice_no_fam_6random_count_ALLS.csv",
                       header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)

count_compliant <- favorite$Answer.manip_check_target == "\"2\"" & favorite$Answer.manip_check_dist == "\"1\""
favorite_wrong <- subset(favorite,!count_compliant)
favorite <- subset(favorite,count_compliant)


favorite$target <- favorite$Answer.choice=="\"target\""
favorite$logical <- favorite$Answer.choice=="\"logical\""
favorite$foil <- favorite$Answer.choice=="\"foil\""

ms.favorite <- aggregate(cbind(target,
                                 logical,
                                 foil) ~ 
                             Answer.item, data=favorite, sum)

es_np_favorite <- mean(favorite$target)

chisq.test(ms.favorite[2:3,2:4])
summary(aov(target ~ as.factor(Answer.item) + as.factor(Answer.target_position), data = favorite))
favorite_wrong$target <- favorite_wrong$Answer.choice=="\"target\""



## With "least favorite"
least_favorite <- read.csv("forced_choice_no_fam_4random_count_12_january_least_LEAS.csv",
                     header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)

count_compliant_least <- least_favorite$Answer.manip_check_target == "\"2\"" & least_favorite$Answer.manip_check_dist == "\"1\""
least_favorite_wrong <- subset(least_favorite,!count_compliant_least)
least_favorite <- subset(least_favorite,count_compliant_least)


least_favorite$target <- least_favorite$Answer.choice=="\"target\""
least_favorite$logical <- least_favorite$Answer.choice=="\"logical\""
least_favorite$foil <- least_favorite$Answer.choice=="\"foil\""

ms.least_favorite <- aggregate(cbind(target,
                               logical,
                               foil) ~ 
                           Answer.item, data=least_favorite, sum)

es_np_least_favorite <- mean(least_favorite$target)

chisq.test(ms.least_favorite[2:3,2:4])
summary(aov(target ~ as.factor(Answer.item) + as.factor(Answer.target_position) + as.factor(Answer.gender), data = least_favorite))
least_favorite_wrong$target <- least_favorite_wrong$Answer.choice=="\"target\""

