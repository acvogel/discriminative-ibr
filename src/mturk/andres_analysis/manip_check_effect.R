library(reshape)
library(ggplot2)

forced_choice_no_fam_6random_count_ALLS.csv
forced_choice_no_fam_6random_NOcount_ALNC.csv


## When the count is requested
with_count <- read.csv("forced_choice_no_fam_6random_count_ALLS.csv",
                    header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)

count_compliant <- with_count$Answer.manip_check_target == "\"2\"" & with_count$Answer.manip_check_dist == "\"1\""
with_count_wrong <- subset(with_count,!count_compliant)
with_count <- subset(with_count,count_compliant)


with_count$target <- with_count$Answer.choice=="\"target\""
with_count$logical <- with_count$Answer.choice=="\"logical\""
with_count$foil <- with_count$Answer.choice=="\"foil\""

ms.with_count <- aggregate(cbind(target,
                         logical,
                         foil) ~ 
                     Answer.item, data=with_count, sum)

es_np_with_count <- mean(with_count$target)

chisq.test(ms.with_count[2:3,2:4])
summary(aov(target ~ as.factor(Answer.item) + as.factor(Answer.target_position), data = with_count))


with_count_wrong$target <- with_count_wrong$Answer.choice=="\"target\""


# When the count is not requested
without_count <- read.csv("forced_choice_no_fam_6random_NOcount_ALNC.csv",
              header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)


without_count$target <- without_count$Answer.choice=="\"target\""
without_count$logical <- without_count$Answer.choice=="\"logical\""
without_count$foil <- without_count$Answer.choice=="\"foil\""

ms.without_count <- aggregate(cbind(target,
                                 logical,
                                 foil) ~ 
                             Answer.item, data=without_count, sum)

es_np_without_count <- mean(without_count$target)

chisq.test(ms.without_count[2:3,2:4])
summary(aov(target ~ as.factor(Answer.item) + as.factor(Answer.target_position), data = without_count))



# Compare side to side
with_and_without = cbind(table(with_count$target), table(without_count$target), table(with_count_wrong$target))
colnames(with_and_without) = c("with", "without", "without_wrong")

chisq.test(with_and_without[,2:3])

table(with_count$Answer.item)
table(without_count$Answer.item)

