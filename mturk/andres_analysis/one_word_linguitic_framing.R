# List of experiments with the one word linguistic framing

forced_choice_no_fam_boat_oneword_7_december_LFBN.csv
forced_choice_no_fam_friend_oneword_1_december_AGSK.csv
forced_choice_no_fam_snowman_oneword_8_december_LSBN.csv
forced_choice_no_fam_sundaes_oneword_8_december_LSBZ.csv


betting_no_fam_friend_oneword_1_december_AGHP.csv
betting_no_fam_boat_oneword_8_december_LZZZ.csv
betting_no_fam_snowman_oneword_8_december_LSSS.csv
betting_no_fam_sundae_oneword_8_december_BSUN.csv

likert_no_fam_friend_oneword_2_december_JSLD.csv
likert_no_fam_snowman_oneword_9_december_FINA.csv
likert_no_fam_sundaes_oneword_9_december_LSUN.csv
likert_no_fam_boat_oneword_9_december_FUNU.csv

library(reshape)
library(ggplot2)


######## FORCED CHOICE
c1 <- read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/forced_choice_no_fam_boat_oneword_7_december_LFBN.csv",
               header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)
c2 <- read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/forced_choice_no_fam_friend_oneword_1_december_AGSK.csv",
               header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)
c3 <- read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/forced_choice_no_fam_snowman_oneword_8_december_LSBN.csv",
               header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)
c4 <- read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/forced_choice_no_fam_sundaes_oneword_8_december_LSBZ.csv",
               header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)
fc = rbind.fill(c1,c2,c3,c4)

fc$compliant <- fc$Answer.name_check_correct == "\"TRUE\""

&
  fc$Answer.manip_check_target == "\"2\"" &
  fc$Answer.manip_check_dist == "\"1\""
  
print(mean(!fc$compliant))

fc <- subset(fc,!compliant)


fc$target <- fc$Answer.choice=="\"target\""
fc$logical <- fc$Answer.choice=="\"logical\""
fc$foil <- fc$Answer.choice=="\"foil\""

ms.fc <- aggregate(cbind(target,
                      logical,
                      foil) ~ 
                  Answer.item, data=fc, sum)

es_np_fc <- mean(fc$target)


######### LIKERT
d1 <- read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/likert_no_fam_sundaes_oneword_9_december_LSUN.csv",
               header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)
d2 <- read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/likert_no_fam_friend_oneword_2_december_JSLD.csv",
               header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)
d3 <- read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/likert_no_fam_snowman_oneword_9_december_FINA.csv",
               header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)
d4 <- read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/likert_no_fam_boat_oneword_9_december_FUNU.csv",
               header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)


likert = rbind.fill(d1,d2,d3, d4)

likert$compliant <- likert$Answer.name_check_correct == "\"TRUE\""
  
&
  likert$Answer.manip_check_target == "\"2\"" &
  likert$Answer.manip_check_dist == "\"1\""
  
print(mean(!likert$compliant))

likert <- subset(likert,!compliant)

qplot(Answer.Likert_value_logical,Answer.Likert_value_target,
      position=position_jitter(width=.2,height=.2),
      colour=Answer.item,
      data=likert)

qplot(Answer.Likert_value_target - Answer.Likert_value_logical,
      fill=Answer.item,
      data=likert)

ms.likert <- aggregate(cbind(Answer.Likert_value_target,
                      Answer.Likert_value_logical,
                      Answer.Likert_value_foil) ~ 
                  Answer.item, data=likert, mean)

es_likert <- mean(likert$Answer.Likert_value_target - likert$Answer.Likert_value_logical) / 
  sd(likert$Answer.Likert_value_target - likert$Answer.Likert_value_logical)

es_np_likert <- mean(likert$Answer.Likert_value_target < likert$Answer.Likert_value_logical)





######## BETTING
b1 <- read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/betting_no_fam_friend_oneword_1_december_AGHP.csv",
               header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)
b2 <- read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/betting_no_fam_boat_oneword_8_december_LZZZ.csv",
               header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)
b3 <- read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/betting_no_fam_snowman_oneword_8_december_LSSS.csv",
               header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)
b4 <- read.csv("/Users/andesgomez/Documents/Stanford/Autumn2013-Masters/PayedWork/andres_data/betting_no_fam_sundae_oneword_8_december_BSUN.csv",
               header=TRUE, sep="\t", row.names=NULL, stringsAsFactors = FALSE)
betting = rbind.fill(b1,b2,b3,b4)

betting$compliant <- betting$Answer.name_check_correct == "\"TRUE\""

&
  betting$Answer.manip_check_target == "\"2\"" &
  betting$Answer.manip_check_dist == "\"1\"" 
  
print(mean(!betting$compliant))

betting <- subset(betting,!compliant)


qplot(Answer.money_allocated_to_logical,Answer.money_allocated_to_target,
      position=position_jitter(width=2,height=2),
      colour=Answer.item,
      data=betting)

qplot(Answer.money_allocated_to_target - Answer.money_allocated_to_logical,
      fill=Answer.item,
      data=betting) + 
  geom_vline(aes(xintercept=mean(Answer.money_allocated_to_target - Answer.money_allocated_to_logical)),
             lty=2)

ms.betting <- aggregate(cbind(Answer.money_allocated_to_target,
                      Answer.money_allocated_to_logical,
                      Answer.money_allocated_to_foil) ~ 
                  Answer.item, data=betting, mean)

es_bet <- mean(betting$Answer.money_allocated_to_target - betting$Answer.money_allocated_to_logical) / 
  sd(betting$Answer.money_allocated_to_target - betting$Answer.money_allocated_to_logical)

es_np_bet <- mean(betting$Answer.money_allocated_to_target > betting$Answer.money_allocated_to_logical)



higher_target_mean = mean(betting[betting$Answer.money_allocated_to_target > betting$Answer.money_allocated_to_logical,]$Answer.money_allocated_to_target)

hist(betting[betting$Answer.money_allocated_to_target > betting$Answer.money_allocated_to_logical,]$Answer.money_allocated_to_target)

