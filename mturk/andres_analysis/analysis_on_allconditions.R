library(stringr)


### Without familiarization
allconditions_no_familiarization = read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/allconditions_no_fam.csv",header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)

forcedchoice_no_fam = allconditions_no_familiarization[allconditions_no_familiarization$Answer.participant_response_type_condition == 0,]
betting_no_fam = allconditions_no_familiarization[allconditions_no_familiarization$Answer.participant_response_type_condition == 1,]
likert_no_fam = allconditions_no_familiarization[allconditions_no_familiarization$Answer.participant_response_type_condition == 2,]


boxplot(betting_no_fam$Answer.money_allocated_to_target, betting_no_fam$Answer.money_allocated_to_logical, betting_no_fam$Answer.money_allocated_to_foil)

hist(betting_no_fam$Answer.money_allocated_to_target)
hist(betting_no_fam$Answer.money_allocated_to_logical)
hist(betting_no_fam$Answer.money_allocated_to_foil)

hist(betting_no_fam$Answer.money_allocated_to_target - betting_no_fam$Answer.money_allocated_to_logical, breaks = 10)
mean(betting_no_fam$Answer.money_allocated_to_target - betting_no_fam$Answer.money_allocated_to_logical)
sd(betting_no_fam$Answer.money_allocated_to_target - betting_no_fam$Answer.money_allocated_to_logical)

fraction_of_forced_choice_correct = sum(forcedchoice_no_fam$Answer.choice_correct == "\"TRUE\"") / length(forcedchoice_no_fam$Answer.choice_correct)

likert_no_fam$Answer.Likert_value_logical = as.numeric(str_sub(likert_no_fam$Answer.Likert_value_logical, 2, 2))
likert_no_fam$Answer.Likert_value_target = as.numeric(str_sub(likert_no_fam$Answer.Likert_value_target, 2, 2))
likert_no_fam$Answer.Likert_value_foil = as.numeric(str_sub(likert_no_fam$Answer.Likert_value_foil, 2, 2))

Likert_col_values = c(35, 51, 49)
relevant_Likert = likert_no_fam[,Likert_col_values]
boxplot(relevant_Likert)
cor(relevant_Likert)
plot(relevant_Likert)
hist(likert_no_fam$Answer.Likert_value_target)
hist(likert_no_fam$Answer.Likert_value_logical)
hist(likert_no_fam$Answer.Likert_value_foil)

coplot(relevant_Likert$Answer.Likert_value_target ~ relevant_Likert$Answer.Likert_value_logical | relevant_Likert$Answer.Likert_value_foil)




### With familiarization
allconditions_yes_familiarization = read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/allconditions_yes_fam.csv",header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)



# Forced choice
forced_choice_yes_fam_from_all = allconditions_yes_familiarization[allconditions_yes_familiarization$Answer.participant_response_type_condition == 0,]

forced_choice_yes_fam_from_all$target = forced_choice_yes_fam_from_all$Answer.choice == "\"target\""
forced_choice_yes_fam_from_all$logical = forced_choice_yes_fam_from_all$Answer.choice == "\"logical\"" 
forced_choice_yes_fam_from_all$foil = forced_choice_yes_fam_from_all$Answer.choice == "\"foil\""

forced_choice_yes_fam_from_all_table <- aggregate(cbind(target,
                                              logical,
                                             foil) ~ 
                                              Answer.familiarization_cond, data=forced_choice_yes_fam_from_all, sum)

ms.ffcc <- mean(forced_choice_yes_fam_from_all$target)

familiarization_variance = glm(target ~ Answer.familiarization_cond, data = forced_choice_yes_fam_from_all)
summary(familiarization_variance)


# Betting
betting_yes_fam_from_all = allconditions_yes_familiarization[allconditions_yes_familiarization$Answer.participant_response_type_condition == 1,]

plot(jitter(betting_yes_fam_from_all$Answer.familiarization_cond, amount = .08), jitter(betting_yes_fam_from_all$Answer.money_allocated_to_target, amount = 0.9), col = "red")
points(jitter(betting_yes_fam_from_all$Answer.familiarization_cond + 0.09, amount = .08), jitter(betting_yes_fam_from_all$Answer.money_allocated_to_logical, amount = 0.9), col = "blue")
points(jitter(betting_yes_fam_from_all$Answer.familiarization_cond + 0.18, amount = .08), jitter(betting_yes_fam_from_all$Answer.money_allocated_to_foil, amount = 0.9), col = "green")

lines(lowess(betting_yes_fam_from_all$Answer.familiarization_cond, betting_yes_fam_from_all$Answer.money_allocated_to_target), col="red") # lowess line (x,y)
lines(lowess(betting_yes_fam_from_all$Answer.familiarization_cond,betting_yes_fam_from_all$Answer.money_allocated_to_logical), col="blue") # lowess line (x,y)
lines(lowess(betting_yes_fam_from_all$Answer.familiarization_cond,betting_yes_fam_from_all$Answer.money_allocated_to_foil), col="green") # lowess line (x,y)

coplot(Answer.money_allocated_to_target ~ Answer.money_allocated_to_logical  | Answer.familiarization_cond, data = betting_yes_fam_from_all)


# Likert
likert_yes_fam_from_all = allconditions_yes_familiarization[allconditions_yes_familiarization$Answer.participant_response_type_condition == 2,]

likert_yes_fam_from_all$Answer.Likert_value_target = (likert_yes_fam_from_all$Answer.Likert_value_target == "\"1\"") * 1 + (likert_yes_fam_from_all$Answer.Likert_value_target == "\"2\"") * 2 + (likert_yes_fam_from_all$Answer.Likert_value_target == "\"3\"") * 3 + (likert_yes_fam_from_all$Answer.Likert_value_target == "\"4\"") * 4 + (likert_yes_fam_from_all$Answer.Likert_value_target == "\"5\"") * 5 +  (likert_yes_fam_from_all$Answer.Likert_value_target == "\"6\"") * 6 + (likert_yes_fam_from_all$Answer.Likert_value_target == "\"7\"") * 7
likert_yes_fam_from_all$Answer.Likert_value_logical = (likert_yes_fam_from_all$Answer.Likert_value_logical == "\"1\"") * 1 + (likert_yes_fam_from_all$Answer.Likert_value_logical == "\"2\"") * 2 + (likert_yes_fam_from_all$Answer.Likert_value_logical == "\"3\"") * 3 + (likert_yes_fam_from_all$Answer.Likert_value_logical == "\"4\"") * 4 + (likert_yes_fam_from_all$Answer.Likert_value_logical == "\"5\"") * 5 +  (likert_yes_fam_from_all$Answer.Likert_value_logical == "\"6\"") * 6 + (likert_yes_fam_from_all$Answer.Likert_value_logical == "\"7\"") * 7
likert_yes_fam_from_all$Answer.Likert_value_foil = (likert_yes_fam_from_all$Answer.Likert_value_foil == "\"1\"") * 1 + (likert_yes_fam_from_all$Answer.Likert_value_foil == "\"2\"") * 2 + (likert_yes_fam_from_all$Answer.Likert_value_foil == "\"3\"") * 3 + (likert_yes_fam_from_all$Answer.Likert_value_foil == "\"4\"") * 4 + (likert_yes_fam_from_all$Answer.Likert_value_foil == "\"5\"") * 5 +  (likert_yes_fam_from_all$Answer.Likert_value_foil == "\"6\"") * 6 + (likert_yes_fam_from_all$Answer.Likert_value_foil == "\"7\"") * 7


plot(jitter(likert_yes_fam_from_all$Answer.familiarization_cond, amount = .08), jitter(likert_yes_fam_from_all$Answer.Likert_value_target, amount = 0.9), col = "red")
points(jitter(likert_yes_fam_from_all$Answer.familiarization_cond + 0.09, amount = .08), jitter(likert_yes_fam_from_all$Answer.Likert_value_logical, amount = 0.9), col = "blue")
points(jitter(likert_yes_fam_from_all$Answer.familiarization_cond + 0.18, amount = .08), jitter(likert_yes_fam_from_all$Answer.Likert_value_foild, amount = 0.9), col = "green")

lines(lowess(likert_yes_fam_from_all$Answer.familiarization_cond, likert_yes_fam_from_all$Answer.Likert_value_target), col="red") # lowess line (x,y)
lines(lowess(likert_yes_fam_from_all$Answer.familiarization_cond,likert_yes_fam_from_all$Answer.Likert_value_logical), col="blue") # lowess line (x,y)
lines(lowess(likert_yes_fam_from_all$Answer.familiarization_cond,likert_yes_fam_from_all$Answer.Likert_value_foil), col="green") # lowess line (x,y)
