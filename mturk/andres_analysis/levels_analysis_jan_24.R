levels <- read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/scale_plus_6stimuli_3levels_no_fam_24_january_SCAL.csv",
                                  header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)

levels$correct <- levels$Answer.choice == "\"target\""

level_0_bool <- levels$Answer.scale_and_levels_condition == 2
level_1_bool <- levels$Answer.scale_and_levels_condition == 3
level_2_bool <- levels$Answer.scale_and_levels_condition == 4

level_0 <- subset(levels, level_0_bool)
level_1 <- subset(levels, level_1_bool)
level_2 <- subset(levels, level_2_bool)

level_0$target <- level_0$Answer.choice=="\"target\""
level_0$logical <- level_0$Answer.choice=="\"logical\""
level_0$foil <- level_0$Answer.choice=="\"foil\""

level_1$target <- level_1$Answer.choice=="\"target\""
level_1$logical <- level_1$Answer.choice=="\"logical\""
level_1$foil <- level_1$Answer.choice=="\"foil\""

level_2$target <- level_2$Answer.choice=="\"target\""
level_2$logical <- level_2$Answer.choice=="\"logical\""
level_2$foil <- level_2$Answer.choice=="\"foil\""

level_0_by_item <- aggregate(cbind(target, logical,  foil) ~ Answer.item, data=level_0, sum)
level_0_pragmatic <- mean(level_0$target)

level_1_by_item <- aggregate(cbind(target, logical,  foil) ~ Answer.item, data=level_1, sum)
level_1_pragmatic <- mean(level_1$target)

level_2_by_item <- aggregate(cbind(target, logical,  foil) ~ Answer.item, data=level_2, sum)
level_2_pragmatic <- mean(level_2$target)

# Printing
level_0_by_item
level_0_pragmatic
level_1_by_item
level_1_pragmatic
level_2_by_item
level_2_pragmatic

sum(level_2_by_item[,2:4])
sum(level_2_by_item[,2])

summary(aov(correct ~ as.factor(Answer.item) * Answer.scale_and_levels_condition, data = levels))
summary(glm(correct ~ as.factor(Answer.item) * as.factor(Answer.scale_and_levels_condition), data = levels))


# Separate by gender and age
levels$males = levels$Answer.gender == "\"m\"" | levels$Answer.gender == "\"male\"" | levels$Answer.gender == "\"M\"" | levels$Answer.gender == "\"Male\"" | levels$Answer.gender == "\"MALE\""
levels$females = levels$Answer.gender == "\"f\"" | levels$Answer.gender == "\"female\"" | levels$Answer.gender == "\"F\"" | levels$Answer.gender == "\"Female\"" | levels$Answer.gender == "\"FEMALE\""

levels$twenties =  levels$Answer.age == "\"20\"" | levels$Answer.age == "\"21\"" | levels$Answer.age == "\"22\"" | levels$Answer.age == "\"23\"" | levels$Answer.age == "\"24\"" | levels$Answer.age == "\"25\"" | levels$Answer.age == "\"26\"" | levels$Answer.age == "\"27\"" | levels$Answer.age == "\"28\"" | levels$Answer.age == "\"29\"" 
levels$thirties = levels$Answer.age == "\"30\"" | levels$Answer.age == "\"31\"" | levels$Answer.age == "\"32\"" | levels$Answer.age == "\"33\"" | levels$Answer.age == "\"34\"" | levels$Answer.age == "\"35\"" | levels$Answer.age == "\"36\"" | levels$Answer.age == "\"37\"" | levels$Answer.age == "\"38\"" | levels$Answer.age == "\"39\"" 
levels$fourties = levels$Answer.age == "\"40\"" | levels$Answer.age == "\"41\"" | levels$Answer.age == "\"42\"" | levels$Answer.age == "\"43\"" | levels$Answer.age == "\"44\"" | levels$Answer.age == "\"45\"" | levels$Answer.age == "\"46\"" | levels$Answer.age == "\"47\"" | levels$Answer.age == "\"48\"" | levels$Answer.age == "\"49\"" 
levels$fifties = levels$Answer.age == "\"50\"" | levels$Answer.age == "\"51\"" | levels$Answer.age == "\"52\"" | levels$Answer.age == "\"53\"" | levels$Answer.age == "\"54\"" | levels$Answer.age == "\"55\"" | levels$Answer.age == "\"56\"" | levels$Answer.age == "\"57\"" | levels$Answer.age == "\"58\"" | levels$Answer.age == "\"59\"" 


summary(glm(correct ~ twenties + thirties + fourties + fifties +
              males + females + 
              as.factor(Answer.item) + as.factor(Answer.scale_and_levels_condition) + as.factor(Answer.target_position), data = levels))




###### Scale ########### Scale ########### Scale ########### Scale ########### Scale ########### Scale ########### Scale #####
levels_scale <- read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/scales_6stimuli_3levels_no_fam_25_january_OSCA.csv",
                   header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)

levels_scale <- levels_scale[1:180,]

levels_scale$correct <- levels_scale$Answer.choice == "\"target\""

level_0_bool_scale <- levels_scale$Answer.scale_and_levels_condition == 0
level_1_bool_scale <- levels_scale$Answer.scale_and_levels_condition == 1

level_0_scale <- subset(levels_scale, level_0_bool_scale)
level_1_scale <- subset(levels_scale, level_1_bool_scale)


level_0_scale$target <- level_0_scale$Answer.choice=="\"target\""
level_0_scale$logical <- level_0_scale$Answer.choice=="\"logical\""
level_0_scale$foil <- level_0_scale$Answer.choice=="\"foil\""

level_1_scale$target <- level_1_scale$Answer.choice=="\"target\""
level_1_scale$logical <- level_1_scale$Answer.choice=="\"logical\""
level_1_scale$foil <- level_1_scale$Answer.choice=="\"foil\""


level_0_by_item_scale <- aggregate(cbind(target, logical,  foil) ~ Answer.item, data=level_0_scale, sum)
level_0_pragmatic_scale <- mean(level_0_scale$target)

level_1_by_item_scale <- aggregate(cbind(target, logical,  foil) ~ Answer.item, data=level_1_scale, sum)
level_1_pragmatic_scale <- mean(level_1_scale$target)


# Printing
level_0_by_item_scale
level_0_pragmatic_scale
level_1_by_item_scale
level_1_pragmatic_scale


sum(level_0_by_item_scale[,2:4])
sum(level_0_by_item_scale[,2])

summary(aov(correct ~ as.factor(Answer.item) * as.factor(Answer.scale_and_levels_condition), data = levels_scale))
summary(glm(correct ~ as.factor(Answer.item) * as.factor(Answer.scale_and_levels_condition), data = levels_scale))


# Separate by gender and age
levels_scale$males = levels_scale$Answer.gender == "\"m\"" | levels_scale$Answer.gender == "\"male\"" | levels_scale$Answer.gender == "\"M\"" | levels_scale$Answer.gender == "\"Male\"" | levels_scale$Answer.gender == "\"MALE\""
levels_scale$females = levels_scale$Answer.gender == "\"f\"" | levels_scale$Answer.gender == "\"female\"" | levels_scale$Answer.gender == "\"F\"" | levels_scale$Answer.gender == "\"Female\"" | levels_scale$Answer.gender == "\"FEMALE\""

levels_scale$twenties =  levels_scale$Answer.age == "\"20\"" | levels_scale$Answer.age == "\"21\"" | levels_scale$Answer.age == "\"22\"" | levels_scale$Answer.age == "\"23\"" | levels_scale$Answer.age == "\"24\"" | levels_scale$Answer.age == "\"25\"" | levels_scale$Answer.age == "\"26\"" | levels_scale$Answer.age == "\"27\"" | levels_scale$Answer.age == "\"28\"" | levels_scale$Answer.age == "\"29\"" 
levels_scale$thirties = levels_scale$Answer.age == "\"30\"" | levels_scale$Answer.age == "\"31\"" | levels_scale$Answer.age == "\"32\"" | levels_scale$Answer.age == "\"33\"" | levels_scale$Answer.age == "\"34\"" | levels_scale$Answer.age == "\"35\"" | levels_scale$Answer.age == "\"36\"" | levels_scale$Answer.age == "\"37\"" | levels_scale$Answer.age == "\"38\"" | levels_scale$Answer.age == "\"39\"" 
levels_scale$fourties = levels_scale$Answer.age == "\"40\"" | levels_scale$Answer.age == "\"41\"" | levels_scale$Answer.age == "\"42\"" | levels_scale$Answer.age == "\"43\"" | levels_scale$Answer.age == "\"44\"" | levels_scale$Answer.age == "\"45\"" | levels_scale$Answer.age == "\"46\"" | levels_scale$Answer.age == "\"47\"" | levels_scale$Answer.age == "\"48\"" | levels_scale$Answer.age == "\"49\"" 
levels_scale$fifties = levels_scale$Answer.age == "\"50\"" | levels_scale$Answer.age == "\"51\"" | levels_scale$Answer.age == "\"52\"" | levels_scale$Answer.age == "\"53\"" | levels_scale$Answer.age == "\"54\"" | levels_scale$Answer.age == "\"55\"" | levels_scale$Answer.age == "\"56\"" | levels_scale$Answer.age == "\"57\"" | levels_scale$Answer.age == "\"58\"" | levels_scale$Answer.age == "\"59\"" 



summary(aov(correct ~ 
              males + females + 
               as.factor(Answer.target_position) + as.factor(Answer.logical_position), data = levels_scale))


total_group = rbind(levels_scale, levels)
sum(levels_scale$females) + sum(levels$females)