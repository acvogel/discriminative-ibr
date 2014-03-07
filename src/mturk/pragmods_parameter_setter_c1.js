// ---------------- 2. STIMULUS SETUP ------------------
// Condition - call the maker getter to get the cond variable 
// Parameters and Stimulus Setup 


// Defining the state-space of relevant experiments for PHASE 0: Methodological questions.

// Participant response type:
//      0 -> NAFC: N alternative forced Choice. In this case N == 3.
//      1 -> Betting: Splitting $100 among three choices (must include validation check)
//      2 -> Likert scale (1 - 7): “how likely is this to be his friend?” 1 = very unlikely, 7 = very likely
//var participant_response_type = random(0,2);
var participant_response_type = 0;

// Participant check trials:
//      0 -> The count of each feature is not asked for
//      1 -> The count of each feature is requested
var participant_feature_count = 1;

// Linguistic framing
//      0 -> "My favorite friend has a hat"
//      1 -> "Bob can only say one word to communicate with you: Hat"
//      2 -> "My least favorite friend has a hat"  - from 2 onwards we try different words to measure the effect of different semantics
//      3 -> "The most beautiful X has a hat"
//      4 -> "The most ugly X has a hat"
//      5 -> "The most cheerful x has a y"
//      6 -> "The most depressing X has y"
//      7 -> Silent favorite: Click below on the option that represents the X that you think is Bob's favorite X.
//      8 -> Silent least favorite: Click below on the option that represents the friend that you think is Bob's least favorite.
var linguistic_framing = 1;

// Question Type (This will be a controlled experiment with an equal proportion for each base rate).
//      0 -> Listener inference judgement
//      1 -> Couldn't hear: “He said ‘My friend has mumblemumble.’ (you didn’t hear what he said)”
//      2 -> Pure base rate: "Which one is his friend?" or "Which friend is Bob's favorite?"
var question_type = 1;

// Question sequence 
//      0 -> Default: One target trial -> probably already done
//      1 -> 6 trials with same inference question
//      2 -> 6 trials, 3 inference, 3 where target is Logical target, unambiguous feature 
//      3 -> Default: One distractor trial  -> probably not necesary
//      4 -> 6 Target Trials
//      5 -> 6 Trials in the order FTFTFT (T=”target trial” F=”filler trial”)
//      6 -> 6 Trials in the order TFTFTF (T=”target trial” F=”filler trial”)
var target_filler_sequence = 0;


// Familiarization Status (whether we have the base rate slide)
//      0 -> We don't have a familiarization stage
//      1 -> We do have a familiarization stage
var familiarization_status = 1;

// The stimulus kind. When it is hardcoded as 1 we get the happy face stimulus.
//    0 -> "boat"
//    1 -> "friend"
//    2 -> "pizza"
//    3 -> "snowman"
//    4 -> "sundae"
//    5 -> "Christmas tree"
var stim_index = random(0,5);
//var stim_index = 2;

// The Scale and Levels.
//    0 -> scales [[0, 0, 0], [0, 0, 1], [0, 1, 1]], level 0
//    1 -> scales [[0, 0, 0], [0, 0, 1], [0, 1, 1]], level 1
//    2 -> scales+ [[0, 0, 1], [0, 1, 1], [1, 1, 0]], level 0
//    3 -> scales+ [[0, 0, 1], [0, 1, 1], [1, 1, 0]], level 1
//    4 -> scales+ [[0, 0, 1], [0, 1, 1], [1, 1, 0]], level 2
//    5 -> scaleweird [[0, 1, 1], [1, 0, 1], [1, 0, 1]] level 0
//    6 -> scaleweird [[0, 1, 1], [1, 0, 1], [1, 0, 1]], level 1
//    7 -> scaleweird [[0, 1, 1], [1, 0, 1], [1, 0, 1]], level 2
var scale_and_level = 1;
//var scale_and_level = 1;

// Elaborate on the purpose of this. Which image is being changed
var img_size = 200; // needs to be implemented, currently just a placeholder   

// Prior familiarization condition
//var cond = 1;
var cond = random(1,4);



// I'm still working on figuring out how exactly this part works,
// but Michael explained that "var condCounts = "1,75;2,75;3,75;4,75""
// is meant to cycle through four conditions (as of yet I don't get)
// which conditions these are supposed to be. 
// http://langcog.stanford.edu/cgi-bin/subject_equalizer/maker_getter.php?conds=1,75;2,75;3,75;4,75&filename=MCF_pragmods_v6
/*
try {
    var filename = "MCF_pragmods_v6"
    var condCounts = "1,75;2,75;3,75;4,75"
    var xmlHttp = null;
    xmlHttp = new XMLHttpRequest();
    xmlHttp.open( "GET", "http://langcog.stanford.edu/cgi-bin/subject_equalizer/maker_getter.php?conds=" + 
		  condCounts +"&filename=" + filename, false );
    xmlHttp.send( null );
    var cond = xmlHttp.responseText;
} catch (e) {
    var cond = 1;
}

*/



// CROSS CONDITIONS
// familiarization conditions. 
//    0 -> Foil, it has neither feature
//    1 -> Target, with the only feature that matters
//    2 -> Logical/distractor, with the two features including the distractor
var fam_dists = [[0, 1, 2, 2, 2, 2, 2, 2, 2],
        [0, 1, 1, 1, 2, 2, 2, 2, 2],
        [0, 1, 1, 1, 1, 1, 2, 2, 2],
        [0, 1, 1, 1, 1, 1, 1, 1, 2]]; 
var target_frequencies = [0.11, 0.33, 0.556, 0.778]; // just counting the proportion of 1s in that matrix, for book keeping and and labeling
// matrix size, usually 9
var instances_in_familiarization = fam_dists[0].length
// matrix size, usually 4
var num_fam_conds = fam_dists.length


// var word_cond = "baserate";  // The type of experiment that is being conducted
var fam_cond = cond - 1; // php is 1 indexed



var fam_dist = fam_dists[fam_cond];

// bookkeeping variables
// All of these are used in the control flow (think about how to organize this)
var choices = [0, 1, 2]; 
var target = -2; // possibly vestigial
var fam_clicked = new Array();   
var fam_finished; // familiarization finished, boolean vareable -- issues with conceptual grouping (think about it)
var positions = ["left","middle","right"];

// ADAPTED FROM PRAGMODS R CODE
var choice_names_unpermuted = ["foil","target","logical"];

// level 3 - what you need to change to change the matrix condition
/*
var expt = [[1, 0, 0], [1, 1, 0], [0, 1, 1]]; // NO SCALES
var level = 2;
var target_unpermuted = 1;
var distractor_unpermuted = 2;
var other_unpermuted = 0;
var target_prop_unpermuted = 1;
var distractor_prop_unpermuted = 0;
*/

// level 1 condition
// m3/r2 and level 1 by default. Then if that is not the case then it is possible to change the variable values

var expt = [[0, 0, 0], [0, 0, 1], [0, 1, 1]]; // SCALES
var level = 1;
var target_unpermuted = 1;
var distractor_unpermuted = 2;
var other_unpermuted = 0;
var target_prop_unpermuted = 2;
var distractor_prop_unpermuted = 1;

if (scale_and_level > 1 && scale_and_level <= 4) {
    expt = [[0, 0, 1], [0, 1, 1], [1, 1, 0]];
}

if (scale_and_level > 4 && scale_and_level <= 7) {
    expt = [[0, 1, 1], [1, 0, 1], [1, 0, 1]];
}

//  Level 0, scales - m2/r3
if (scale_and_level == 0) {
    var level = 0;
    var target_unpermuted = 2;
    var distractor_unpermuted = 1;
    var other_unpermuted = 0;
    var target_prop_unpermuted = 1;
    var distractor_prop_unpermuted = 2;
    var foil_prop_unpermuted = 0;
    var choice_names_unpermuted = ["foil","logical","target"];
}

//  Level 1, scales - m3/r2
if (scale_and_level == 1) {
    var level = 1;
    var target_unpermuted = 1;
    var distractor_unpermuted = 2;
    var other_unpermuted = 0;
    var target_prop_unpermuted = 2;
    var distractor_prop_unpermuted = 1;
    var foil_prop_unpermuted = 0;
    var choice_names_unpermuted = ["foil","target","logical"];
}

// Level 0, scales+ - m1/r3     
// Consider replacing "logical" for a second "foil", since neither alternative is strictly coherent
if (scale_and_level == 2) {
    var level = 0;
    var target_unpermuted = 2;
    var distractor_unpermuted = 1;
    var other_unpermuted = 0;
    var target_prop_unpermuted = 0;
    var distractor_prop_unpermuted = 1;
    var foil_prop_unpermuted = 2;
    var choice_names_unpermuted = ["foil","logical","target"];
}
// Level 1, scales+ - m3/r1
if (scale_and_level == 3) {
    var level = 1;
    var target_unpermuted = 0;
    var distractor_unpermuted = 1;
    var other_unpermuted = 0;
    var target_prop_unpermuted = 2;
    var distractor_prop_unpermuted = 1;
    var foil_prop_unpermuted = 0;
    var choice_names_unpermuted = ["target","logical","foil"];
}

// Level 2, scales+ - m2/r2
if (scale_and_level == 4) {
    var level = 2;
    var target_unpermuted = 1;
    var distractor_unpermuted = 2;
    var other_unpermuted = 0;
    var target_prop_unpermuted = 1;
    var distractor_prop_unpermuted = 0;
    var foil_prop_unpermuted = 2;
    var choice_names_unpermuted = ["foil","target","logical"];
}

// Compltetely underdetermined because the target feature is everywehre.
if (scale_and_level == 5) {
    var level = 0;
    var target_unpermuted = 0;
    var distractor_unpermuted = 1;
    var other_unpermuted = 2;
    var target_prop_unpermuted = 1;
    var distractor_prop_unpermuted = 0;
    var foil_prop_unpermuted = 2;
    var choice_names_unpermuted = ["single","twin","twin"];
}

if (scale_and_level == 6) {
    var level = 1;
    var target_unpermuted = 1;
    var distractor_unpermuted = 2;
    var other_unpermuted = 0;
    var target_prop_unpermuted = 0;
    var distractor_prop_unpermuted = 1;
    var foil_prop_unpermuted = 2;
    var choice_names_unpermuted = ["single","twin","twin"];
}

if (scale_and_level == 7) {
    var level = 2;
    var target_unpermuted = 1;
    var distractor_unpermuted = 2;
    var other_unpermuted = 0;
    var target_prop_unpermuted = 2;
    var distractor_prop_unpermuted = 1;
    var foil_prop_unpermuted = 0;
    var choice_names_unpermuted = ["single","twin","twin"];
}



var stims = ["boat","friend","pizza","snowman","sundae","Christmas tree"];
var stims_plural = ["boats","friends","pizzas","snowmen","sundaes","Christmas trees"];

var stims_props = [["cabin","sail","motor"],
		   ["hat","glasses","mustache"],
		   ["mushrooms","olives","peppers"],
                   ["hat","scarf","mittens"],		   
                   ["cherry","whipped cream","chocolate"],
		   ["lights","ornaments","star"]];
var stims_prop_words = [["a cabin","a sail","a motor"],
			["a hat","glasses","a mustache"],
			["mushrooms","olives","peppers"],
                        ["a hat","a scarf","mittens"],
                        ["a cherry","whipped cream","chocolate sauce"],
			["lights","ornaments","a star"]];
var stims_single_words = [["cabin","sail","motor"],
            ["hat","glasses","mustache"],
            ["mushrooms","olives","peppers"],
            ["hat","scarf","mittens"],
            ["cherry","whipped-cream","chocolate"],
            ["lights","ornaments","star"]];
var stims_actions = [["sail","rents","sailed", "rented"],
		     ["visit","chooses to visit","visited", "visited"],
		     ["eat","orders","ate", "eaten"],
		     ["decorate","makes", "decorated", "decorated"],
		     ["eat","makes","ate", "eaten"],
		     ["trim","buys","trimmed", "trimmed"]];
var stims_times = [["weekend","Week"],
		   ["Sunday","Week"],
		   ["Wednesday","Week"],
		   ["winter","Year"],
		   ["Friday","Week"],
		   ["Christmas","Year"]];


// In case the number of possible stimulus is changed and hence the stim_index may vary. So far we know it advance there
// are only 6 stimulus.
// var stim_index = random(0,stims.length-1);



// Permute the matrix randomly:
var prop_perm = shuffle(range(0,expt[0].length-1));
var target_perm = shuffle(range(0,expt.length-1));
var expt_perm = new Array();
var choice_names = new Array();


// Permute the second level and first... make into function
for (var i=0; i<expt.length; i++) {
    expt_perm[i] = new Array()
    for (var j=0; j<expt[0].length; j++) {
    	expt_perm[i][j] = expt[target_perm[i]][prop_perm[j]];
    }
    choice_names[i] = choice_names_unpermuted[target_perm[i]];
}

var base = stims[stim_index];
var plural = stims_plural[stim_index];
var actions = stims_actions[stim_index];
var props = stims_props[stim_index];
var prop_words = stims_prop_words[stim_index];
var individual_prop_words = stims_single_words[stim_index];
var times = stims_times[stim_index];

var target = target_perm.indexOf(target_unpermuted);
var distractor = target_perm.indexOf(distractor_unpermuted);
var other = target_perm.indexOf(other_unpermuted);
var target_prop = prop_perm.indexOf(target_prop_unpermuted);
var distractor_prop = prop_perm.indexOf(distractor_prop_unpermuted);
var foil_prop = prop_perm.indexOf(foil_prop_unpermuted);

var actual_target_prop = prop_words[target_prop];
var actual_distractor_prop = prop_words[distractor_prop];

// create shuffled familiarization
fam_mat = new Array();
fam_perm = shuffle(range(0,fam_dist.length-1))
for (var i = 0; i < instances_in_familiarization; i++) {
    fam_mat[i] = new Array();
    for (var j = 0; j < expt[0].length; j++) {
    	fam_mat[i][j] = expt[fam_dist[fam_perm[i]]][prop_perm[j]];
    }
}
