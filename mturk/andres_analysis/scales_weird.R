favorite_weird <- read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/scaleweird_6stimuli_no_fam_favorite_19_february_WERF.csv",
                      header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)

favorite_weird = subset(favorite_weird, favorite_weird$Answer.name_check_correct == "\"TRUE\"") # Name check correct

favorite_weird$twin <- favorite_weird$Answer.choice== "\"twin\""
favorite_weird$single <- favorite_weird$Answer.choice=="\"single\""

level_1 <- subset(favorite_weird, favorite_weird$Answer.scale_and_levels_condition == 6)
level_2 <- subset(favorite_weird, favorite_weird$Answer.scale_and_levels_condition == 7)

favorite_weird_randomized <- aggregate(cbind(twin,
                                        single) ~ 
                                         Answer.target_position, data=favorite_weird, sum)

ms.favorite_weird <- mean(favorite_weird$twin)


item_variance = aov(twin ~ as.factor(Answer.item) + as.factor(Answer.scale_and_levels_condition) + as.factor(Answer.logical_position) + as.factor(Answer.target_position), data = favorite_weird)
summary(item_variance)



# One word:
oneword_weird <- read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/scaleweird_6stimuli_no_fam_oneword_19_february_WERD.csv",
                           header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)



oneword_weird = subset(oneword_weird, oneword_weird$Answer.name_check_correct == "\"TRUE\"") # Name check correct

oneword_weird$twin <- oneword_weird$Answer.choice== "\"twin\""
oneword_weird$single <- oneword_weird$Answer.choice=="\"single\""

oneword_weird$distractor_position = (oneword_weird$Answer.target_position == "\"left\"") * -1 + (oneword_weird$Answer.target_position == "\"right\"") * 1 + (oneword_weird$Answer.logical_position == "\"left\"") * -1 + (oneword_weird$Answer.logical_position == "\"right\"") * 1
 

level_1 <- subset(oneword_weird, oneword_weird$Answer.scale_and_levels_condition == 6)
level_2 <- subset(oneword_weird, oneword_weird$Answer.scale_and_levels_condition == 7)

oneword_weird_randomized <- aggregate(cbind(twin,
                                             single) ~ 
                                        Answer.item + Answer.scale_and_levels_condition, data=oneword_weird, sum)


average_twin_level_1 <- sum(level_1$twin); average_twin_level_1
average_twin_level_2 <- length(level_2$twin); average_twin_level_2


variance_analysis = aov(twin ~ as.factor(Answer.item) + as.factor(distractor_position), data = level_2)
summary(variance_analysis)

logistic_analysis = glm(twin ~ as.factor(Answer.scale_and_levels_condition) + as.factor(Answer.item) + as.factor(distractor_position), data = oneword_weird)
summary(logistic_analysis)


binom.test(40, 67, 0.75)

level_1$Answer.comment