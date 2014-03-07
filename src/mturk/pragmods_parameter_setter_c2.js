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
var linguistic_framing = 1;

// Question Type (This will be a controlled experiment with an equal proportion for each base rate).
//      0 -> Listener inference judgement
//      1 -> Couldn't hear: “He said ‘My friend has mumblemumble.’ (you didn’t hear what he said)”
//      2 -> Pure base rate: "Which one is his friend?" or "Which friend is Bob's favorite?"
var question_type = 0;

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
var familiarization_status = 0;

// The stimulus kind. When it is hardcoded as 1 we get the happy face stimulus.
//    0 -> "boat"
//    1 -> "friend"
//    2 -> "pizza"
//    3 -> "snowman"
//    4 -> "sundae"
//    5 -> "Christmas tree"
var stim_index = random(0,5);
//var stim_index = 2;

// Elaborate on the purpose of this. Which image is being changed
var img_size = 200; // needs to be implemented, currently just a placeholder   

// Prior familiarization condition
var cond = 1;
// var cond = random(1,4);



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
var target_frequencies = [0.22, 0.44, 0.667, 0.89]; // just counting the proportion of 1s in that matrix, for book keeping and and labeling
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

var expt = [[0, 0, 0], [0, 0, 1], [0, 1, 1]]; // SCALES
var level = 1;
var target_unpermuted = 1;
var distractor_unpermuted = 2;
var other_unpermuted = 0;
var target_prop_unpermuted = 2;
var distractor_prop_unpermuted = 1;


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
// When target_filler_sequence is not equal to zero:
stimuli_to_show = range(0, 6);
ordered_stimuli = shuffle(stimuli_to_show);
stim_index = ordered_stimuli[0]




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