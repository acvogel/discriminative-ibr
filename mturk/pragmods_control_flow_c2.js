
// ---------------- 3. CONTROL FLOW ------------------
// This .js file determines the flow of the variable elements in the experiment as dictated 
// by the various calls from pragmods html.

/*
Here the images used in the experiment are loaded in two arrays.
The first is base_image_pl, which stores the "underlying" or base images
which will then be modified with props stored in the props_image_pl Array.

NOTE: Unfortunately the number of variations for each type of object is hardoded.
To make the code more usable it will be necessary to 
*/

var base_image_pl = new Array();
for (i=0; i<3; i++) {
    base_image_pl[i] = new Image();
    base_image_pl[i].src = "images2/" + base + "-base" + String(i+1) + ".png";
}


// By creating image object and setting source, the proloaded images become accesible. In this case they are stored as elements of the Array.
var props_image_pl = new Array() 
for (i=0;i<props.length;i++) {
    props_image_pl[i] = new Image();
    props_image_pl[i].src = "images2/" + base + "-" + props[i] + ".png";
}

var number_to_name = new Array();
number_to_name[0] = 'A';
number_to_name[1] = 'B';
number_to_name[2] = 'C';


// The variable that counts the number of trials done.
trials_completed = 0;
choice_sequence = ["null", "null", "null", "null", "null", "null"]

showSlide("instructions");

// The main experiment:
//		The variable/object 'experiment' has two distinct but interrelated components:
//		1) It collects the variables that will be delivered to Turk at the end of the experiment
//		2) It hall all the functions that change the visual aspect of the experiment such as showing images, etc.

var experiment = {

    // These variables are the ones that will be sent to Turk at the end.
    // The first batch, however, is set determined by the experiment conditions
    // and therefore should not be affected by the decisions made by the experimental subject.
	item: base,
	target_property: props[target_prop],
	target_frequency: target_frequencies[fam_cond],
	target_position: positions[target],    // -2
	familiarization_cond: fam_cond, // This is the index number of the familiarization conditions. For example, fam_cond == 0 means that the distractors, targets etc. are: [0, 1, 2, 2, 2, 2, 2, 2, 2]
	
	// These variables define the state-space of the experiment. linguistic_framing is subordinated to question_type (only matters for the first question type.)
	participant_response_type_condition: participant_response_type,
	participant_feature_count_condition: participant_feature_count,
	linguistic_framing_condition: linguistic_framing,
	target_filler_sequence_condition: target_filler_sequence,
	question_type_condition: question_type,

	// These are the variables that will be modified as a function of how the user responds to the experiment.
	// They are the ones necessary to do any science at all. 
	manip_check_target: -1,  // -1 is the default, which means that the subject did not enter any number because it was not asked to count (participant_feature_count == 1)
	manip_check_dist: -1,
	name_check_correct: "FALSE",

	// The relevant variables when the participatn response type is: Forced Choice
	choice: "null",
	choice_correct: "null",

	// The relevant variables when the participant response type is: Betting
	// The allocated money goes to: "foil","target" and "logical" and it must sum to $100
	money_allocated_to_foil: -1,
	money_allocated_to_target: -1,
	money_allocated_to_logical: -1,


	// The relevant variables when the participant response type is: Slider/Likert scale
	Likert_value_foil: -1,
	Likert_value_target: -1,
	Likert_value_logical: -1,

	// Free form text given by the participant
	about: "",
	comment: "",

	// For my first batch, I'll make it so that this represents whether familiarization was present or not... 1 is not, -1 is yes.
	//debugVariable: -1,

	familiarization_present_in_study: familiarization_status,


	// When target_filler_sequence is not 0
	choice_1: "null",
	choice_2: "null",
	choice_3: "null",
	choice_4: "null",
	choice_5: "null",
	choice_6: "null",


	// FAMILIARIZATION DISPLAY FUNCTION
	// This 
	next_familiarization: function() {
	    // Allow experiment to start if it's a turk worker OR if it's a test run
	    // Warning, it may not be security best practices... you know why
		if (window.self == window.top | turk.workerId.length > 0) {

			// When there is a familiarization stage that sets up the priors.
			if (familiarization_status == 1) {
				// FAMILIARIZATION INSTRUCTIONS
				var familiarization_html = '<p class="block-text"">Bob really likes to ' + 
				actions[0] + ' ' + plural + '. <br>' +
				'Every ' + times[0] + ' he ' + actions[1] + ' a ' + base + '.<p>' +
				'<p class="block-text">Click on each ' + times[1].toLowerCase() + ' to see the ' + base + ' Bob ' + actions[2] + '.</p>';
				$("#familiarizationText").html(familiarization_html);

				// TIME BY TIME POPUPS FOR FAMILIARIZATION
				// Here are all the different elements that will be displayed in the familiarization task
				// most likely it will always be 9, but not necessarily
				// That is why we have a table here
				var familiarization_objects_html = '<table align="center"><tr>';

				for (i=0;i<=instances_in_familiarization-1;i++){
					familiarization_objects_html += '<td width=200px height=230px align="center" ' +
						'class="objTable"' + 
						'id="famTable' + String(i) + 
						'" onclick="experiment.reveal(' + String(i) + ')">';
					familiarization_objects_html += '<br><br><br><br>' +times[1] + ' ' + String(i+1) + ', Bob ' + actions[2] 
						+ ':<div id="day' + String(i) + '"> </div>';
					if ((i+1)%3 == 0) {
						familiarization_objects_html += "</tr><tr>";
					}
				}
				familiarization_objects_html += '</tr></table>';
				$("#famObjects").html(familiarization_objects_html);
			} else {
				experiment.target_frequency = 0;
				experiment.familiarization_cond = -1;
				// Instructions when there is no familiarization: Presenting Bob and explaning what that Bob does.
				// The importance of this slide in this condition (familiarization_status == 0) is to reiffy 
				// the social situation. In other words, make it clear that *someone* is asking you about the person.
				var familiarization_html = '<p class="block-text"">Bob really likes to ' + 
				actions[0] + ' ' + plural + '. <br>' + 
				'Every ' + times[0] + ' he ' + actions[1] + ' a ' + base + '.<p>';
				$("#familiarizationText").html(familiarization_html);
			}
			showSlide("prestage");	
		}
	},

	// MAIN DISPLAY FUNCTION
	next_test: function() {

		showSlide("stage");

		// CREATE SETUP
		var setup_html = '<p class="block-text">Take a look at these ' + plural + ' Bob has ' + actions[3] + '!</p>';
		$("#setup").html(setup_html);

		// CREATE OBJECT TABLE
		// The key thing here is that forced_choice_objects_html is something that will be *posted* on the HTML 
		// So if we make alternative objects depending on the value of *participant_response_type*
		// (tr=table row; td= table data)
		// When participant_response_type == 0
		if (participant_response_type == 0) {
			var forced_choice_objects_html = '<table align="center"><tr>';
			// Q: Is this 3 a variable thing 
			for (i=0;i<3;i++) {
				forced_choice_objects_html += '<td width=198px height=210px align="center"' + 
				' class="notChoices objTable">';
				forced_choice_objects_html += stimHTML(base,i,expt_perm[i],props,'obj');
				forced_choice_objects_html += '</td>';
			}
			forced_choice_objects_html += '</tr><tr>';
			for (i = 0; i < 3; i++) {
				forced_choice_objects_html += '<td width=198px height=20px align="center"' + 
				' class="notChoices objTable" ' +
				'id=\"tdchoice' + String(i+3) + '\">';
				forced_choice_objects_html += ' ' + number_to_name[i] + ' ';
				forced_choice_objects_html += '</td>';
			}
			forced_choice_objects_html += '</tr><tr></tr></table><br><br>';
			$("#objects").html(forced_choice_objects_html) 
		}


		// Betting Participant Input Type
		// When participant_response_type == 1
		if (participant_response_type == 1) {
			var betting_amounts_object_html = '<table align="center"><tr>';
			for (i=0;i<3;i++) {
				betting_amounts_object_html += '<td width=198px height=210px align="center"' + 
				' class="notChoices objTable" ' +
				'id=\"tdchoice' + String(i) + '\">';
				betting_amounts_object_html += stimHTML(base,i,expt_perm[i],props,'obj');
				betting_amounts_object_html += '</td>';
			}
			betting_amounts_object_html += '</tr><tr>';
			for (i = 0; i < 3; i++) {
				betting_amounts_object_html += '<td width=198px height=20px align="center"' + 
				' class="notChoices objTable" ' +
				'id=\"tdchoice' + String(i+3) + '\">';
				betting_amounts_object_html += ' ' + number_to_name[i] + ' ';
				betting_amounts_object_html += '</td>';
			}
			betting_amounts_object_html += '</tr><tr></tr></table><br><br>';
			$("#objects").html(betting_amounts_object_html);
		}

		// Likert option (fancy slider)
		// When participant_response_type == 2
		if (participant_response_type == 2) {
			var Likert_object_html = '<table align="center"><tr>';
			for (i=0;i<3;i++) {
				Likert_object_html += '<td width=198px height=210px align="center"' + 
				' class="notChoices objTable" ' +
				'id=\"tdchoice' + String(i) + '\">';
				Likert_object_html += stimHTML(base,i,expt_perm[i],props,'obj');
				Likert_object_html += '</td>';
			}
			Likert_object_html += '</tr><tr>';
			for (i = 0; i < 3; i++) {
				Likert_object_html += '<td width=198px height=30px align="center"' + 
				' class="notChoices objTable" ' +
				'id=\"tdchoice' + String(i+3) + '\">';
				Likert_object_html += ' ' + number_to_name[i] + ' ';
				Likert_object_html += '</td>';
			}
			Likert_object_html += '</tr></table><br><br>';
			$("#objects").html(Likert_object_html) 
		}		


		// CREATE MANIPULATION CHECK COMMON GROUNDING
		// Here we ask for the count of each of the features.

		if (participant_feature_count == 1) {
			var manipCheck_html = "";

			checkTarget = '<p class="block-text">How many of the ' + plural + ' have ' + 
				actual_target_prop + '?' + '  <input type="text" id="manipCheckTarget" ' + 
				'name="manipCheckTarget" size="1"></p>';
			checkDistractor = '<p class="block-text">How many of the ' + plural + ' have ' + 
				actual_distractor_prop + '?' + 
				'  <input type="text" id="manipCheckDist" ' + 
				'name="manipCheckDist" size="1"></p>';

			// randomize the order... this previously just had the lines above randomly added to manipCheck_html. This is cleaner.
			if (random(0,1) == 0) { 
				manipCheck_html += checkTarget;
				manipCheck_html += checkDistractor;
			} else {
				manipCheck_html += checkDistractor;
				manipCheck_html += checkTarget;
			} 
			$("#manipCheck").html(manipCheck_html)
		}


		// Create the way in which the question is asked (and the type of question)
		var label_html = '<br><br><p class="block-text">';
		if (question_type == 0) { // LISTENER CONDITION
			if (linguistic_framing == 0) {
				label_html += 'Bob says: ';
		    	label_html += '<p class="block-text style="font-size:x-large;">' + '"My favorite ' + base + ' has <b>' + prop_words[target_prop] + '."</b></p>';
			} else if (linguistic_framing == 1) {
				label_html += 'Bob can only say one word to communicate with you and he says: ';
				label_html += '<p class="block-text style="font-size:x-large;">  <b>' + individual_prop_words[target_prop] + '</b></p>';
			}
		} else if (question_type == 1) { // SCHELLING (MUMBLE) CONDITION
			label_html += 'Bob says: '
		    label_html += '<p class="block-text style="font-size:x-large;">' + '"The ' + base + ' I most like to ' + actions[0] + ' has <b>mumblemumble."</b></p>' + '<p class="block-text style="font-size:small;">' + '(You couldn\'t hear what he said.)</p>';
		} else if (question_type == 2) { // PURE BASE RATE CONDITION
			label_html += 'Which ' + base + ' is Bob\'s favorite?</p>';
		}

		// Explain what the user is supposed to do for the pragmatic inference

		if (linguistic_framing == 0) {
			if (participant_response_type == 0) {
				label_html += '<p class="block-text">Click below on the option that represents the ' + base + ' that you think is Bob\'s favorite.</p>';
			} else if (participant_response_type == 1) {
				label_html += '<p class="block-text">You have $100 you can use to bet on the friends you think may be Bob\'s favorite. Distribute your $100 among the options by how likely you think that each of the options is Bob\'s favorite. (Make sure your bets add to $100).</p>';
			} else if (participant_response_type == 2) {
				label_html += '<p class="block-text">On a scale from 1 to 7, for each ' + base + ' choose the level of confidence that you have that it is Bob\'s favorite. Here 1 means "very confident that it is not his favorite", 7 means "very confident that it is his favorite" and 4 means that you are not sure one way or the other.</p>';
			}
		} else if (linguistic_framing == 1) {
			if (participant_response_type == 0) {
				label_html += '<p class="block-text">Click below on the option that represents the ' + base + ' that you think Bob is talking about.</p>';
			} else if (participant_response_type == 1) {
				label_html += '<p class="block-text">You have $100 you can use to bet on the ' + base + ' you think Bob is talking about. Distribute your $100 among the options by how likely you think that Bob is referring to each of the options. (Make sure your bets add to $100).</p>';
			} else if (participant_response_type == 2) {
				label_html += '<p class="block-text">On a scale from 1 to 7, for each ' + base + ' choose the level of confidence that you have that Bob is referring to it. Here 1 means "very confident that Bob is NOT referring to it", 7 means "very confident that Bob is referring to it" and 4 means that you are not sure one way or the other.</p>';
			}
		}


		$("#labelInst").html(label_html) 	

		// Here add the code that matters for the experiment. Note: the input fields about the pragmatic inference
		// used to be just underneath the stimuli. This got changed to force the subjects to first read the instructions
		// and only then answer the questions.
		var user_input_selection = '';
		if (participant_response_type == 0) {
			user_input_selection += '<table align="center"><tr>';
			for (i=0;i<3;i++) {
				user_input_selection += '<td width=98px height=50px align="center"' + 
					' class="unchosen objTable" ' +
					'id="tdchoice' + String(i) + '" ' +
					'onclick=\"experiment.select(' + String(i) + ')\">';
				user_input_selection +=  '<br>' + number_to_name[i];
				user_input_selection += '</td>';
			}
			user_input_selection += '</tr><tr>';
			user_input_selection += '</tr></table>';
		} else if (participant_response_type == 1) {
			user_input_selection += '<table align="center"><tr>';
			for (i = 0; i < 3; i++) {
				user_input_selection += '<td width=198px height=30px align="center"' + 
				' class="notChoices objTable" ' +
				'id=\"tdchoice' + String(i+3) + '\">';
				user_input_selection += '$<input type="text" id="betForOption' + String(i) + '" name="betForOption' + String(i) + '" size="4"> for ' + number_to_name[i];
				user_input_selection += '</td>';
			}
			user_input_selection += '</tr><tr></tr></table>';
		} else if (participant_response_type == 2) {
			user_input_selection += '<br><table align="center"><tr>';
			for (i = 0; i < 3; i++) {
			user_input_selection += '<td width=198px height=30px align="center"' + 
				' class="notChoices objTable" ' +
				'id=\"tdchoice' + String(i+3) + '\">';
				user_input_selection += 'Level of confidence for ' + number_to_name[i];
				user_input_selection += '</td>';
			}
			user_input_selection += '</tr><tr>';
			for (i = 0; i < 3; i++) {
				user_input_selection += '<td width=198px height=30px align="center"' + 
				' class="notChoices objTable" ' +
				'id=\"tdchoice' + String(i+6) + '\">';
				user_input_selection += '1 <input type="radio" id="likertFor' + String(i) + '_1' + '" name="likertFor' + String(i) + '" size="1" value = "1">';
				user_input_selection += '<input type="radio" id="likertFor' + String(i) + '_2' + '" name="likertFor' + String(i) + '" size="1" value = "2">';
				user_input_selection += '<input type="radio" id="likertFor' + String(i) + '_3' + '" name="likertFor' + String(i) + '" size="1" value = "3">';
				user_input_selection += '<input type="radio" id="likertFor' + String(i) + '_4' + '" name="likertFor' + String(i) + '" size="1" value = "4">';
				user_input_selection += '<input type="radio" id="likertFor' + String(i) + '_5' + '" name="likertFor' + String(i) + '" size="1" value = "5">';
				user_input_selection += '<input type="radio" id="likertFor' + String(i) + '_6' + '" name="likertFor' + String(i) + '" size="1" value = "6">';
				user_input_selection += '<input type="radio" id="likertFor' + String(i) + '_7' + '" name="likertFor' + String(i) + '" size="1" value = "7"> 7';
				user_input_selection += '</td>';
			}
			user_input_selection += '</tr><tr></tr></table>';
		}

		$("#userSelectionInputFields").html(user_input_selection)
	},

		// SELECT FUNCTION (called in stage slide)
	select: function (c) {
		experiment.choice = choice_names[c];

		// unchoose everything
		for (var i=0; i<choices.length; i++) {
		    $("#tdchoice" + String(i)).removeClass('chosen').addClass('unchosen')
		}
		// choose this one
		$("#tdchoice" + String(c)).removeClass('unchosen').addClass('chosen')

		},

		// REVEAL IMAGES IN FAMILIARIZATION
	reveal: function(n) {
			day_html = stimHTML(base,fam_dist[fam_perm[n]],fam_mat[n],props,'obj')
			$("#day" + String(n)).html(day_html) 
			fam_clicked = unique(fam_clicked.concat(n))

			if (fam_clicked.length == instances_in_familiarization) {
			    fam_finished = 1
			}
		},

	// CHECK THAT FAMILIARIZARION IS DONE
	check_fam: function() {
		if (fam_finished == 1 || familiarization_status == 0) {
		    famNextButton.blur(); 
		    experiment.next_test();
		} else {
		    $("#famMessage").html('<font color="red">' + 
				       'Please make sure you have looked at all the '+ base +  '!' + 
				       '</font>');
		}
    },

    // CHECK THAT TEST IS DONE
    // It does not ask for the numeric input from the subject if the participant is not asked to count.
    // The condition is different depending on the type of response (input expected by the user)
    check_test: function() {

    	// This variable just takes care of the case where the subject IS asked to count the features.
    	// That is, its value is 0 when the requirements needed when the counts are needed are not in place.
    	// And 1 in all other cases. And when participant is asked to count, this also assigns
    	// the variables manip_check_target and manip_check_dist as needed.
    	var count_condition_fulfilled = 0;
    	if (participant_feature_count == 0) {
    		count_condition_fulfilled = 1;
    	} else if (document.getElementById("manipCheckTarget").value != "" && document.getElementById("manipCheckDist").value != "") {
    		count_condition_fulfilled = 1;
    		experiment.manip_check_target = document.getElementById("manipCheckTarget").value;
	    	experiment.manip_check_dist = document.getElementById("manipCheckDist").value;
    	}

    	// Checking that IF we have a forced choice type then the conditions are fulfilled
    	var forced_choice_condition_fulfilled = 0;
    	if (participant_response_type != 0 || experiment.choice != "null") {
    		forced_choice_condition_fulfilled = 1;
    	}

    	// Checking that IF we have a betting settup, the bets work well
    	var betting_condition_fulfilled = 0;
    	if (participant_response_type != 1) {
    		betting_condition_fulfilled = 1;
    	} else {
    		for (var i = 0; i <3; i ++) {
    			if (choice_names[i] == "foil") {
    				experiment.money_allocated_to_foil =  parseInt(document.getElementById("betForOption" + String(i)).value);
    			} else if (choice_names[i] == "target") {
    				experiment.money_allocated_to_target = parseInt(document.getElementById("betForOption" + String(i)).value);
    			} else if (choice_names[i] == "logical") {
    				experiment.money_allocated_to_logical = parseInt(document.getElementById("betForOption" + String(i)).value);
    			}
    		}
    		var sumationOverBets = experiment.money_allocated_to_foil + experiment.money_allocated_to_target + experiment.money_allocated_to_logical;
    		if (sumationOverBets == 100 &&  experiment.money_allocated_to_foil >= 0 
    			&& experiment.money_allocated_to_target >= 0 && experiment.money_allocated_to_logical >= 0 
    			&& experiment.money_allocated_to_foil <= 100 && experiment.money_allocated_to_target <= 100 
    			&& experiment.money_allocated_to_logical <= 100) {
    			betting_condition_fulfilled = 1;
    		}
    	}

    	// Checking that the Likert scale options work.
    	var likert_condition_fulfilled = 0;
    	if (participant_response_type != 2) {
    		likert_condition_fulfilled = 1;
    	} else {
    		for (var i = 0; i <3; i ++) {

    			var listOfLikerOptions = ["likertFor" + String(i) + "_1", "likertFor" + String(i) + "_2", "likertFor" + String(i) + "_3", "likertFor" + String(i) + "_4", "likertFor" + String(i) + "_5", "likertFor" + String(i) + "_6", "likertFor" + String(i) + "_7"];

    			if (choice_names[i] == "foil") {
    				experiment.Likert_value_foil =  parseInt(getNameRadioValue(listOfLikerOptions));
    			} else if (choice_names[i] == "target") {
    				experiment.Likert_value_target = parseInt(getNameRadioValue(listOfLikerOptions));
    			} else if (choice_names[i] == "logical") {
    				experiment.Likert_value_logical = parseInt(getNameRadioValue(listOfLikerOptions));
    			}
    		}
    		var testRadio = 1;
    		if (experiment.Likert_value_foil == 1 || experiment.Likert_value_foil == 2 || experiment.Likert_value_foil == 3 || experiment.Likert_value_foil == 4 || experiment.Likert_value_foil == 5 || experiment.Likert_value_foil == 6 ||experiment.Likert_value_foil == 7) {
    		} else {
    			testRadio = 0;
    		}
    		if (experiment.Likert_value_target == 1 || experiment.Likert_value_target == 2 || experiment.Likert_value_target == 3 || experiment.Likert_value_target == 4 || experiment.Likert_value_target == 5 || experiment.Likert_value_target == 6 ||experiment.Likert_value_target == 7) {
    		} else {
    			testRadio = 0;
    		}
    		if (experiment.Likert_value_logical == 1 || experiment.Likert_value_logical == 2 || experiment.Likert_value_logical == 3 || experiment.Likert_value_logical == 4 || experiment.Likert_value_logical == 5 || experiment.Likert_value_logical == 6 ||experiment.Likert_value_logical == 7) {
    		} else {
    			testRadio = 0;
    		}
    		if (testRadio == 1) {
    			likert_condition_fulfilled = 1;
    		}
    	}

    	if (forced_choice_condition_fulfilled == 1 && betting_condition_fulfilled == 1 && likert_condition_fulfilled == 1 && count_condition_fulfilled == 1) {
    		// Here you repeat the experiment if the trial number is not done.
    		trials_completed = trials_completed + 1;
    		if (trials_completed < 6) {
				stim_index = ordered_stimuli[trials_completed]

				base = stims[stim_index];
				plural = stims_plural[stim_index];
				actions = stims_actions[stim_index];
				props = stims_props[stim_index];
				prop_words = stims_prop_words[stim_index];
				individual_prop_words = stims_single_words[stim_index];
				times = stims_times[stim_index];

				target = target_perm.indexOf(target_unpermuted);
				distractor = target_perm.indexOf(distractor_unpermuted);
				other = target_perm.indexOf(other_unpermuted);
				target_prop = prop_perm.indexOf(target_prop_unpermuted);
				distractor_prop = prop_perm.indexOf(distractor_prop_unpermuted);

				actual_target_prop = prop_words[target_prop];
				actual_distractor_prop = prop_words[distractor_prop];

				testNextButton.blur(); 
				experiment.next_familiarization();
    		} else {
				testNextButton.blur(); 
				showSlide("check");
    		}
		} else {
	    	$("#testMessage").html('<font color="red">' + 
			'Please make sure you have answered all the questions!' + 
			'</font>');
			if (betting_condition_fulfilled == 0){
				$("#testMessage").html('<font color="red">' + 
				'Please make sure that all your bets add to $100 and that you have given a value to each option' + 
				'</font>');
			}
			if (likert_condition_fulfilled == 0) {
				$("#testMessage").html('<font color="red">' + 
				'Please make sure you have given an answer to each option!' + 
				'</font>');
			}
		}
    },

   // FINISHED BUTTON CHECKS EVERYTHING AND THEN ENDS
    check_finished: function() {
    	var listOfNameRadios = ["namecheck1", "namecheck2", "namecheck3", "namecheck4"];
    	personMet = getNameRadioValue(listOfNameRadios);
		if (personMet == 0 ||
		    document.getElementById('about').value.length < 1) {
		    $("#checkMessage").html('<font color="red">' + 
				       'Please make sure you have answered all the questions!' + 
				       '</font>');
		} else {
		    if (personMet == 1) {
				experiment.name_check_correct = "TRUE";
		    }
		    experiment.about = document.getElementById("about").value;
		    experiment.comment = document.getElementById("comments").value;
		    showSlide("finished");

		    // HERE you can performe the needed boolean logic to properly account for the target_filler_sequence possibilities.
		    // In other words, here you can check whether the choice is correct depending on the nature of the trial.

		    if (experiment.choice == "target") {
				experiment.choice_correct = "TRUE";
		    } else {
		    	experiment.choice_correct = "FALSE";
		    }
		    experiment.end();
		}
    },

    // END FUNCTION 
    end: function () {
    	showSlide("finished");
    	setTimeout(function () {
		turk.submit(experiment);
        }, 500); 
    }
}