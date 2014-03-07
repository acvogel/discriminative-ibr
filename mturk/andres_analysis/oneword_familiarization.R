### With familiarization
familiarization = read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/scale_6stimuli_yes_fam_oneword_25_february_FAMO.csv",header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)

familiarization$target = familiarization$Answer.choice == "\"target\""
familiarization$logical = familiarization$Answer.choice == "\"logical\"" 
familiarization$foil = familiarization$Answer.choice == "\"foil\""

#familiarization = subset(familiarization, familiarization$Answer.name_check_correct == "\"TRUE\"")

fc_oneword_table <- aggregate(cbind(target,
                                        logical,
                                        foil) ~ 
                                        Answer.familiarization_cond, data=familiarization, mean)

ms.ffcc <- mean(familiarization$target)



familiarization$males = familiarization$Answer.gender == "\"m\"" | familiarization$Answer.gender == "\"male\"" | familiarization$Answer.gender == "\"M\"" | familiarization$Answer.gender == "\"Male\"" | familiarization$Answer.gender == "\"MALE\""
familiarization$females = familiarization$Answer.gender == "\"f\"" | familiarization$Answer.gender == "\"female\"" | familiarization$Answer.gender == "\"F\"" | familiarization$Answer.gender == "\"Female\"" | familiarization$Answer.gender == "\"FEMALE\""

familiarization$twenties =  familiarization$Answer.age == "\"20\"" | familiarization$Answer.age == "\"21\"" | familiarization$Answer.age == "\"22\"" | familiarization$Answer.age == "\"23\"" | familiarization$Answer.age == "\"24\"" | familiarization$Answer.age == "\"25\"" | familiarization$Answer.age == "\"26\"" | familiarization$Answer.age == "\"27\"" | familiarization$Answer.age == "\"28\"" | familiarization$Answer.age == "\"29\"" 
familiarization$thirties = familiarization$Answer.age == "\"30\"" | familiarization$Answer.age == "\"31\"" | familiarization$Answer.age == "\"32\"" | familiarization$Answer.age == "\"33\"" | familiarization$Answer.age == "\"34\"" | familiarization$Answer.age == "\"35\"" | familiarization$Answer.age == "\"36\"" | familiarization$Answer.age == "\"37\"" | familiarization$Answer.age == "\"38\"" | familiarization$Answer.age == "\"39\"" 
familiarization$fourties = familiarization$Answer.age == "\"40\"" | familiarization$Answer.age == "\"41\"" | familiarization$Answer.age == "\"42\"" | familiarization$Answer.age == "\"43\"" | familiarization$Answer.age == "\"44\"" | familiarization$Answer.age == "\"45\"" | familiarization$Answer.age == "\"46\"" | familiarization$Answer.age == "\"47\"" | familiarization$Answer.age == "\"48\"" | familiarization$Answer.age == "\"49\"" 
familiarization$fifties = familiarization$Answer.age == "\"50\"" | familiarization$Answer.age == "\"51\"" | familiarization$Answer.age == "\"52\"" | familiarization$Answer.age == "\"53\"" | familiarization$Answer.age == "\"54\"" | familiarization$Answer.age == "\"55\"" | familiarization$Answer.age == "\"56\"" | familiarization$Answer.age == "\"57\"" | familiarization$Answer.age == "\"58\"" | familiarization$Answer.age == "\"59\"" 





# Analysis of variance and Regression

single_group_variance = aov(target ~ as.factor(Answer.familiarization_cond) + as.factor(Answer.item), data = familiarization)
summary(single_group_variance)

familiarization_variance = glm(target ~ as.factor(Answer.familiarization_cond) + as.factor(Answer.item) + as.factor(Answer.target_position) + as.factor(Answer.logical_position), data = familiarization)
summary(familiarization_variance)


familiarization_control = aov(target ~ as.factor(Answer.familiarization_cond) + as.factor(Answer.item) + as.factor(females) + as.factor(males) + twenties + fifties + fourties, data = familiarization)
summary(familiarization_control)

manip_check_dist
manip_check_target
name_check_correct


familiarization$Answer.comment