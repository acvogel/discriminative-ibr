
# Favorite Friend
forced_choice_no_fam_friend_favorite = read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/forced_choice_no_fam_friends_30_november.csv",header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)
sum(forced_choice_no_fam_friend_favorite$Answer.choice == "\"target\"")

forced_choice_no_fam_friend_favorite$target <- forced_choice_no_fam_friend_favorite$Answer.choice=="\"target\""
forced_choice_no_fam_friend_favorite$logical <- forced_choice_no_fam_friend_favorite$Answer.choice=="\"logical\""
forced_choice_no_fam_friend_favorite$foil <- forced_choice_no_fam_friend_favorite$Answer.choice=="\"foil\""

ms.favorite_friend <- aggregate(cbind(target,
                               logical,
                               foil) ~ 
                           Answer.item, data=forced_choice_no_fam_friend_favorite, sum)

# Favorite boat
forced_choice_no_fam_boat_favorite = read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/forced_choice_no_fam_boat_1_december_XYZQ_incomplete.csv",header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)
sum(forced_choice_no_fam_boat_favorite$Answer.choice == "\"target\"")

forced_choice_no_fam_boat_favorite$target <- forced_choice_no_fam_boat_favorite$Answer.choice=="\"target\""
forced_choice_no_fam_boat_favorite$logical <- forced_choice_no_fam_boat_favorite$Answer.choice=="\"logical\""
forced_choice_no_fam_boat_favorite$foil <- forced_choice_no_fam_boat_favorite$Answer.choice=="\"foil\""

ms.favorite_boat <- aggregate(cbind(target,
                                      logical,
                                      foil) ~ 
                                  Answer.item, data=forced_choice_no_fam_boat_favorite, sum)

# Favorite pizza
forced_choice_no_fam_pizza_favorite = read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/forced_choice_no_fam_pizza_30_november.csv",header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)
sum(forced_choice_no_fam_pizza_favorite$Answer.choice == "\"target\"")


forced_choice_no_fam_pizza_favorite$target <- forced_choice_no_fam_pizza_favorite$Answer.choice=="\"target\""
forced_choice_no_fam_pizza_favorite$logical <- forced_choice_no_fam_pizza_favorite$Answer.choice=="\"logical\""
forced_choice_no_fam_pizza_favorite$foil <- forced_choice_no_fam_pizza_favorite$Answer.choice=="\"foil\""

ms.favorite_pizza <- aggregate(cbind(target,
                                    logical,
                                    foil) ~ 
                                Answer.item, data=forced_choice_no_fam_pizza_favorite, sum)

# Favorite snowman
forced_choice_no_fam_snowman_favorite = read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/forced_choice_no_fam_snowman_30_november.csv",header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)
sum(forced_choice_no_fam_snowman_favorite$Answer.choice == "\"target\"")

forced_choice_no_fam_snowman_favorite$target <- forced_choice_no_fam_snowman_favorite$Answer.choice=="\"target\""
forced_choice_no_fam_snowman_favorite$logical <- forced_choice_no_fam_snowman_favorite$Answer.choice=="\"logical\""
forced_choice_no_fam_snowman_favorite$foil <- forced_choice_no_fam_snowman_favorite$Answer.choice=="\"foil\""

ms.favorite_snowman <- aggregate(cbind(target,
                                    logical,
                                    foil) ~ 
                                Answer.item, data=forced_choice_no_fam_snowman_favorite, sum)


total_table_favorite = rbind(ms.favorite_boat, ms.favorite_friend, ms.favorite_pizza, ms.favorite_snowman)
ms.favorite = sum(total_table_favorite$target) / sum(total_table_favorite[1:4,2:4])

favorite_separated_together = rbind(forced_choice_no_fam_friend_favorite, forced_choice_no_fam_snowman_favorite, forced_choice_no_fam_pizza_favorite, forced_choice_no_fam_boat_favorite)

##################################################################################################################


# With "least favorite"
least_favorite <- read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/forced_choice_no_fam_4random_count_12_january_least_LEAS.csv",
                           header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)

count_compliant_least <- least_favorite$Answer.manip_check_target == "\"2\"" & least_favorite$Answer.manip_check_dist == "\"1\""
least_favorite_wrong <- subset(least_favorite,!count_compliant_least)
least_favorite <- subset(least_favorite,count_compliant_least)


least_favorite$target <- least_favorite$Answer.choice=="\"target\""
least_favorite$logical <- least_favorite$Answer.choice=="\"logical\""
least_favorite$foil <- least_favorite$Answer.choice=="\"foil\""

least_favorite_randomized <- aggregate(cbind(target,
                                     logical,
                                     foil) ~ 
                                 Answer.item, data=least_favorite, sum)

ms.es_np_least_favorite <- mean(least_favorite$target)


##################################################################################################################


# With new linguistic framing of one word   -> The large and complete (but random kinds)
one_word_lf <- read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/forced_choice_no_fam_6random_count_ALLS.csv",
                       header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)

count_compliant <- one_word_lf$Answer.manip_check_target == "\"2\"" & one_word_lf$Answer.manip_check_dist == "\"1\""
one_word_lf_wrong <- subset(one_word_lf,!count_compliant)
one_word_lf <- subset(one_word_lf,count_compliant)

one_word_lf$target <- one_word_lf$Answer.choice=="\"target\""
one_word_lf$logical <- one_word_lf$Answer.choice=="\"logical\""
one_word_lf$foil <- one_word_lf$Answer.choice=="\"foil\""

total_one_word_randomized <- aggregate(cbind(target,
                                 logical,
                                 foil) ~ 
                             Answer.item, data=one_word_lf, sum)

ms.es_one_word <- mean(one_word_lf$target)

summary(aov(target ~ as.factor(Answer.item) * as.factor(Answer.target_position),data = one_word_lf))


##################################################################################################################

# One word friend

one_word_friend = read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/forced_choice_no_fam_friend_oneword_1_december_AGSK.csv",header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)
sum(one_word_friend$Answer.choice == "\"target\"")

one_word_friend$target <- one_word_friend$Answer.choice=="\"target\""
one_word_friend$logical <- one_word_friend$Answer.choice=="\"logical\""
one_word_friend$foil <- one_word_friend$Answer.choice=="\"foil\""

ms.one_word_friend <- aggregate(cbind(target,
                                      logical,
                                      foil) ~ 
                                  Answer.item, data=one_word_friend, sum)


# One word boat

one_word_boat = read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/forced_choice_no_fam_boat_oneword_7_december_LFBN_2.csv",header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)
sum(one_word_boat$Answer.choice == "\"target\"")

one_word_boat$target <- one_word_boat$Answer.choice=="\"target\""
one_word_boat$logical <- one_word_boat$Answer.choice=="\"logical\""
one_word_boat$foil <- one_word_boat$Answer.choice=="\"foil\""

ms.one_word_boat <- aggregate(cbind(target,
                                      logical,
                                      foil) ~ 
                                  Answer.item, data=one_word_boat, sum)

# One word snowman
one_word_snowman = read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/forced_choice_no_fam_snowman_oneword_8_december_LSBN.csv",header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)
sum(one_word_snowman$Answer.choice == "\"target\"")

one_word_snowman$target <- one_word_snowman$Answer.choice=="\"target\""
one_word_snowman$logical <- one_word_snowman$Answer.choice=="\"logical\""
one_word_snowman$foil <- one_word_snowman$Answer.choice=="\"foil\""

ms.one_word_snowman <- aggregate(cbind(target,
                                      logical,
                                      foil) ~ 
                                  Answer.item, data=one_word_snowman, sum)


# One word sundae
one_word_sundae = read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/forced_choice_no_fam_sundaes_oneword_8_december_LSBZ.csv",header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)
sum(one_word_sundae$Answer.choice == "\"target\"")

one_word_sundae$target <- one_word_sundae$Answer.choice=="\"target\""
one_word_sundae$logical <- one_word_sundae$Answer.choice=="\"logical\""
one_word_sundae$foil <- one_word_sundae$Answer.choice=="\"foil\""

ms.one_word_sundae <- aggregate(cbind(target,
                                      logical,
                                      foil) ~ 
                                  Answer.item, data=one_word_sundae, sum)

total_table_one_word = rbind(ms.one_word_friend, ms.one_word_boat, ms.one_word_snowman, ms.one_word_sundae)
ms.one_word_total = sum(total_table_one_word$target) / sum(total_table_one_word[1:4,2:4])

one_word_separated_together = rbind(one_word_sundae, one_word_snowman, one_word_boat, one_word_friend)


########################################################################################################


# Asking for what Bob thinks is the most beautiful
most_beautiful <- read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/forced_choice_no_fam_6random_count_beautiful_15_january_BEAU.csv",
                        header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)

count_compliant <- most_beautiful$Answer.manip_check_target == "\"2\"" & most_beautiful$Answer.manip_check_dist == "\"1\""
most_beautiful_wrong <- subset(most_beautiful,!count_compliant)
most_beautiful <- subset(most_beautiful,count_compliant)

most_beautiful$target <- most_beautiful$Answer.choice=="\"target\""
most_beautiful$logical <- most_beautiful$Answer.choice=="\"logical\""
most_beautiful$foil <- most_beautiful$Answer.choice=="\"foil\""

most_beautiful_randomized <- aggregate(cbind(target,
                                             logical,
                                             foil) ~ 
                                         Answer.item, data=most_beautiful, sum)

ms.most_beautiful <- mean(most_beautiful$target)


################### This one is wrong #########

most_ugly_wrong <- read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/forced_choice_no_fam_6random_count_ugly_15_january_UHGG.csv",
                           header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)

count_compliant <- most_ugly_wrong$Answer.manip_check_target == "\"2\"" & most_ugly_wrong$Answer.manip_check_dist == "\"1\""
most_ugly_wrong_www <- subset(most_ugly_wrong,!count_compliant)
most_ugly_wrong <- subset(most_ugly_wrong,count_compliant)

most_ugly_wrong$target <- most_ugly_wrong$Answer.choice=="\"target\""
most_ugly_wrong$logical <- most_ugly_wrong$Answer.choice=="\"logical\""
most_ugly_wrong$foil <- most_ugly_wrong$Answer.choice=="\"foil\""

most_ugly_wrong_randomized <- aggregate(cbind(target,
                                             logical,
                                             foil) ~ 
                                         Answer.item, data=most_ugly_wrong, sum)

ms.most_ugly_wrong <- mean(most_ugly_wrong$target)



################### Ugly done right ###############

most_ugly <- read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/forced_choice_no_fam_6random_count_ugly_15_january_UGLY.csv",
                            header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)

count_compliant <- most_ugly$Answer.manip_check_target == "\"2\"" & most_ugly$Answer.manip_check_dist == "\"1\""
most_ugly_wrong_www <- subset(most_ugly,!count_compliant)
most_ugly <- subset(most_ugly,count_compliant)

most_ugly$target <- most_ugly$Answer.choice=="\"target\""
most_ugly$logical <- most_ugly$Answer.choice=="\"logical\""
most_ugly$foil <- most_ugly$Answer.choice=="\"foil\""

most_ugly_randomized <- aggregate(cbind(target,
                                        logical,
                                        foil) ~ 
                                    Answer.item, data=most_ugly, sum)

ms.most_ugly <- mean(most_ugly$target)


################### Cheerful ###############

most_cheerful <- read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/forced_choice_no_fam_6random_count_cheerful_17_january_CHEE.csv",
                      header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)

count_compliant_cheerful <- most_cheerful$Answer.manip_check_target == "\"2\"" & most_cheerful$Answer.manip_check_dist == "\"1\""
most_cheerful_non_compliant <- subset(most_cheerful,!count_compliant_cheerful)
most_cheerful <- subset(most_cheerful,count_compliant_cheerful)

most_cheerful$target <- most_cheerful$Answer.choice=="\"target\""
most_cheerful$logical <- most_cheerful$Answer.choice=="\"logical\""
most_cheerful$foil <- most_cheerful$Answer.choice=="\"foil\""

most_cheerful_randomized <- aggregate(cbind(target,
                                        logical,
                                        foil) ~ 
                                    Answer.item, data=most_cheerful, sum)

ms.most_cheerful <- mean(most_cheerful$target)

################### Depressing ###############


most_depressing <- read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/forced_choice_no_fam_6random_count_depressing_17_january_DEPR.csv",
                          header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)

count_compliant_depressing <- most_depressing$Answer.manip_check_target == "\"2\"" & most_depressing$Answer.manip_check_dist == "\"1\""
most_depressing_non_compliant <- subset(most_depressing,!count_compliant_depressing)
most_depressing <- subset(most_depressing,count_compliant_depressing)

most_depressing$target <- most_depressing$Answer.choice=="\"target\""
most_depressing$logical <- most_depressing$Answer.choice=="\"logical\""
most_depressing$foil <- most_depressing$Answer.choice=="\"foil\""

most_depressing_randomized <- aggregate(cbind(target,
                                            logical,
                                            foil) ~ 
                                        Answer.item, data=most_depressing, sum)

ms.most_depressing <- mean(most_depressing$target)

################# Base Rate "favorite" Bob is silent #################

silent_favorite <- read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/forced_choice_no_fam_6random_count_favorite_22_january_MUTE.csv",
                            header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)

count_compliant_silent_favorite <- silent_favorite$Answer.manip_check_target == "\"2\"" & silent_favorite$Answer.manip_check_dist == "\"1\""
silent_favorite_non_compliant <- subset(silent_favorite,!count_compliant_silent_favorite)
silent_favorite <- subset(silent_favorite,count_compliant_silent_favorite)

silent_favorite$target <- silent_favorite$Answer.choice=="\"target\""
silent_favorite$logical <- silent_favorite$Answer.choice=="\"logical\""
silent_favorite$foil <- silent_favorite$Answer.choice=="\"foil\""

silent_favorite_randomized <- aggregate(cbind(target,
                                              logical,
                                              foil) ~ 
                                          Answer.item, data=silent_favorite, sum)

ms.silent_favorite <- mean(silent_favorite$target)


################# Base Rate "least favorite" Bob is silent #################

silent_least_favorite <- read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/forced_choice_no_fam_6random_count_least_favorite_22_january_MUTL.csv",
                            header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)

count_compliant_silent_least_favorite <- silent_least_favorite$Answer.manip_check_target == "\"2\"" & silent_least_favorite$Answer.manip_check_dist == "\"1\""
silent_least_favorite_non_compliant <- subset(silent_least_favorite,!count_compliant_silent_least_favorite)
silent_least_favorite <- subset(silent_least_favorite,count_compliant_silent_least_favorite)

silent_least_favorite$target <- silent_least_favorite$Answer.choice=="\"target\""
silent_least_favorite$logical <- silent_least_favorite$Answer.choice=="\"logical\""
silent_least_favorite$foil <- silent_least_favorite$Answer.choice=="\"foil\""

silent_least_favorite_randomized <- aggregate(cbind(target,
                                              logical,
                                              foil) ~ 
                                          Answer.item, data=silent_least_favorite, sum)

ms.silent_least_favorite <- mean(silent_least_favorite$target)



################### ALL OF The RESUTS #########

# Favorite 
total_table_favorite
ms.favorite
# Least favorite 
least_favorite_randomized
ms.es_np_least_favorite
# One word - all at once
total_one_word_randomized
ms.es_one_word
# One word - separated
total_table_one_word
ms.one_word_total
# Beautiful
most_beautiful_randomized
ms.most_beautiful
# Ugly Wrong
most_ugly_wrong_randomized
ms.most_ugly_wrong
# Ugly - no mistake here
most_ugly_randomized
ms.most_ugly
# Cheerful
most_cheerful_randomized
ms.most_cheerful
# Depressing
most_depressing_randomized
ms.most_depressing
# Bob's favorite without Bob saying anything
silent_favorite_randomized
ms.silent_favorite
# Bob's least favorite without Bob saying anything
silent_least_favorite_randomized
ms.silent_least_favorite

####################################### SemFest writeup #######################################

# Brief comparisons of the different silent conditions.
sum(total_table_favorite[1:4,3]) / sum(total_table_favorite[1:4,2:4])
sum(least_favorite_randomized[1:4,3]) / sum(least_favorite_randomized[1:4,2:4])

sum(silent_least_favorite_randomized[1:4,3]) / sum(silent_least_favorite_randomized[1:4,2:4])
sum(silent_favorite_randomized[1:4,4]) / sum(silent_favorite_randomized[1:4,2:4])

silent_least_favorite$response = silent_least_favorite$logical*2 + silent_least_favorite$target*1
silent_favorite$response = silent_favorite$logical*2 + silent_favorite$target*1
favorite_separated_together$response = favorite_separated_together$logical*2 + favorite_separated_together$target*1
least_favorite$response = least_favorite$logical*2 + least_favorite$target*1

summary(glm(response ~ as.factor(Answer.item) + as.factor(Answer.target_position), data = silent_least_favorite))
summary(glm(response ~ as.factor(Answer.item) + as.factor(Answer.target_position), data = silent_favorite))
summary(glm(response ~ as.factor(Answer.item) + as.factor(Answer.target_position), data = favorite_separated_together))
summary(glm(response ~ as.factor(Answer.item) + as.factor(Answer.target_position), data = least_favorite))


############## Brief Michael's analysis ####################
#To get the model predictions, you are trying to compute p(r | w, C), the probability of the referent given the word and the context, which by bayes rule is

#p( w | r, C) * p(r | C)

#divided by the sum of this across all referents. So for the level 1 implicature, for favorite for the target, this is something like

#p("glasses" | target) * .205  /  (p("glasses" | no-item) * .11 + p("glasses" | target) * .205 + p("glasses" | logical) * .685)

#where p("glasses" | target) = (1/ |"glasses"|) / (1/ |"glasses"|) = 1
#and for logical it's (1/ |"glasses"|) / (1/ |"glasses"| + 1/ |"hat"|)) = (1/2) / ((1/2) + 1) = (1/2) / (3/2) = 1/3
#and for no-item it's p("glasses" | no-item) = 0

#So that gives us

#p("glasses" | target) * .22  /  (p("glasses" | no-item) * .13 + p("glasses" | target) * .22 + p("glasses" | logical) * .65)
#
.22 / (.22 + (1/3) * .65)

.14 / (.14 + (1/3)*.21)

#which is actually not bad as a prediction! (Compare to the actual result, .53, below). - I am writing fast and haven't checked this but it seems reasonable.

# For least favorite: p( w | r, C) * p(r | C) = 
#P("glasses" | target ) * 















######################### GLOBAL ANALYSIS OF VARIANCE AND PREDICTORS ################################


one_word_separated_together$Answer.age = 1:length(one_word_separated_together$target) * 0
one_word_separated_together$Answer.gender = 1:length(one_word_separated_together$target) * 0

favorite_separated_together$Answer.age = 1:length(favorite_separated_together$target) * 0
favorite_separated_together$Answer.gender = 1:length(favorite_separated_together$target) * 0

one_word_lf$Answer.age = 1:length(one_word_lf$target) * 0
one_word_lf$Answer.gender = 1:length(one_word_lf$target) * 0

global_pool = rbind(silent_favorite, most_depressing, most_cheerful, most_ugly, most_beautiful, least_favorite, one_word_separated_together, favorite_separated_together, one_word_lf)

# Sanitize Gener and Age
global_pool$males = global_pool$Answer.gender == "\"m\"" | global_pool$Answer.gender == "\"male\"" | global_pool$Answer.gender == "\"M\"" | global_pool$Answer.gender == "\"Male\"" | global_pool$Answer.gender == "\"MALE\""
global_pool$females = global_pool$Answer.gender == "\"f\"" | global_pool$Answer.gender == "\"female\"" | global_pool$Answer.gender == "\"F\"" | global_pool$Answer.gender == "\"Female\"" | global_pool$Answer.gender == "\"FEMALE\""

global_pool$twenties =  global_pool$Answer.age == "\"20\"" | global_pool$Answer.age == "\"21\"" | global_pool$Answer.age == "\"22\"" | global_pool$Answer.age == "\"23\"" | global_pool$Answer.age == "\"24\"" | global_pool$Answer.age == "\"25\"" | global_pool$Answer.age == "\"26\"" | global_pool$Answer.age == "\"27\"" | global_pool$Answer.age == "\"28\"" | global_pool$Answer.age == "\"29\"" 
global_pool$thirties = global_pool$Answer.age == "\"30\"" | global_pool$Answer.age == "\"31\"" | global_pool$Answer.age == "\"32\"" | global_pool$Answer.age == "\"33\"" | global_pool$Answer.age == "\"34\"" | global_pool$Answer.age == "\"35\"" | global_pool$Answer.age == "\"36\"" | global_pool$Answer.age == "\"37\"" | global_pool$Answer.age == "\"38\"" | global_pool$Answer.age == "\"39\"" 
global_pool$fourties = global_pool$Answer.age == "\"40\"" | global_pool$Answer.age == "\"41\"" | global_pool$Answer.age == "\"42\"" | global_pool$Answer.age == "\"43\"" | global_pool$Answer.age == "\"44\"" | global_pool$Answer.age == "\"45\"" | global_pool$Answer.age == "\"46\"" | global_pool$Answer.age == "\"47\"" | global_pool$Answer.age == "\"48\"" | global_pool$Answer.age == "\"49\"" 
global_pool$fifties = global_pool$Answer.age == "\"50\"" | global_pool$Answer.age == "\"51\"" | global_pool$Answer.age == "\"52\"" | global_pool$Answer.age == "\"53\"" | global_pool$Answer.age == "\"54\"" | global_pool$Answer.age == "\"55\"" | global_pool$Answer.age == "\"56\"" | global_pool$Answer.age == "\"57\"" | global_pool$Answer.age == "\"58\"" | global_pool$Answer.age == "\"59\"" 
  

# ANOVA
anova_effects = aov(target ~ as.factor(Answer.item) * as.factor(Answer.linguistic_framing_condition) + as.factor(Answer.target_position)  + as.factor(Answer.name_check_correct) + males + females + twenties + thirties + fourties + fifties, data = global_pool)
summary(anova_effects)

anova_subtle = aov(target ~ as.factor(Answer.target_position) + as.factor(Answer.name_check_correct) + as.factor(Answer.target_property) + males + females, data = global_pool)
summary(anova_subtle)




# Regression
regression_effects = glm(target ~ as.factor(Answer.item) + as.factor(Answer.linguistic_framing_condition)  + as.factor(Answer.target_position) + males + females + twenties + thirties + fourties + fifties, data = global_pool)
summary(regression_effects)

subtle_regression = glm(target ~ Answer.age + Answer.gender, data = global_pool)
summary(subtle_regression)


regresion_lineal_simple = function (datos_como_matriz)
{
  return ( lm( formula = as.formula(paste(colnames(datos_como_matriz), collapse='~') ), data = as.data.frame(datos_como_matriz)))
}