### With familiarization_mumble
familiarization_mumble = read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/scale_6stimuli_yes_fam_mumblemumble_26_february_FMMM.csv",header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)

familiarization_mumble$target = familiarization_mumble$Answer.choice == "\"target\""
familiarization_mumble$logical = familiarization_mumble$Answer.choice == "\"logical\"" 
familiarization_mumble$foil = familiarization_mumble$Answer.choice == "\"foil\""

#familiarization_mumble = subset(familiarization_mumble, familiarization_mumble$Answer.name_check_correct == "\"TRUE\"")

fc_mumble_table <- aggregate(cbind(target,
                                     logical,
                                     foil) ~ 
                                 Answer.familiarization_cond, data=familiarization_mumble, mean)

mean_target <- mean(familiarization_mumble$target)



familiarization_mumble$males = familiarization_mumble$Answer.gender == "\"m\"" | familiarization_mumble$Answer.gender == "\"male\"" | familiarization_mumble$Answer.gender == "\"M\"" | familiarization_mumble$Answer.gender == "\"Male\"" | familiarization_mumble$Answer.gender == "\"MALE\""
familiarization_mumble$females = familiarization_mumble$Answer.gender == "\"f\"" | familiarization_mumble$Answer.gender == "\"female\"" | familiarization_mumble$Answer.gender == "\"F\"" | familiarization_mumble$Answer.gender == "\"Female\"" | familiarization_mumble$Answer.gender == "\"FEMALE\""

familiarization_mumble$twenties =  familiarization_mumble$Answer.age == "\"20\"" | familiarization_mumble$Answer.age == "\"21\"" | familiarization_mumble$Answer.age == "\"22\"" | familiarization_mumble$Answer.age == "\"23\"" | familiarization_mumble$Answer.age == "\"24\"" | familiarization_mumble$Answer.age == "\"25\"" | familiarization_mumble$Answer.age == "\"26\"" | familiarization_mumble$Answer.age == "\"27\"" | familiarization_mumble$Answer.age == "\"28\"" | familiarization_mumble$Answer.age == "\"29\"" 
familiarization_mumble$thirties = familiarization_mumble$Answer.age == "\"30\"" | familiarization_mumble$Answer.age == "\"31\"" | familiarization_mumble$Answer.age == "\"32\"" | familiarization_mumble$Answer.age == "\"33\"" | familiarization_mumble$Answer.age == "\"34\"" | familiarization_mumble$Answer.age == "\"35\"" | familiarization_mumble$Answer.age == "\"36\"" | familiarization_mumble$Answer.age == "\"37\"" | familiarization_mumble$Answer.age == "\"38\"" | familiarization_mumble$Answer.age == "\"39\"" 
familiarization_mumble$fourties = familiarization_mumble$Answer.age == "\"40\"" | familiarization_mumble$Answer.age == "\"41\"" | familiarization_mumble$Answer.age == "\"42\"" | familiarization_mumble$Answer.age == "\"43\"" | familiarization_mumble$Answer.age == "\"44\"" | familiarization_mumble$Answer.age == "\"45\"" | familiarization_mumble$Answer.age == "\"46\"" | familiarization_mumble$Answer.age == "\"47\"" | familiarization_mumble$Answer.age == "\"48\"" | familiarization_mumble$Answer.age == "\"49\"" 
familiarization_mumble$fifties = familiarization_mumble$Answer.age == "\"50\"" | familiarization_mumble$Answer.age == "\"51\"" | familiarization_mumble$Answer.age == "\"52\"" | familiarization_mumble$Answer.age == "\"53\"" | familiarization_mumble$Answer.age == "\"54\"" | familiarization_mumble$Answer.age == "\"55\"" | familiarization_mumble$Answer.age == "\"56\"" | familiarization_mumble$Answer.age == "\"57\"" | familiarization_mumble$Answer.age == "\"58\"" | familiarization_mumble$Answer.age == "\"59\"" 





# Analysis of variance and Regression

single_group_variance = aov(logical ~ as.factor(Answer.target_frequency) + as.factor(Answer.item), data = familiarization_mumble)
summary(single_group_variance)

familiarization_mumble_variance = aov(target ~ as.factor(Answer.familiarization_cond) + as.factor(Answer.item) + as.factor(Answer.target_position) + as.factor(Answer.logical_position), data = familiarization_mumble)
summary(familiarization_mumble_variance)


familiarization_mumble_control = aov(target ~ as.factor(Answer.familiarization_cond) + as.factor(Answer.item) + as.factor(females) + as.factor(males) + twenties + fifties + fourties, data = familiarization_mumble)
summary(familiarization_mumble_control)

manip_check_dist
manip_check_target
name_check_correct