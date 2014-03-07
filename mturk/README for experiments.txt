** pragmods_js: new javascript ecosystem for pragmods experiments **

-------  v1: initial test, trying things out -------

------- v2: stiller scales -------

first run at stiller scales

------- v3: stiller noscales modified, requires > L1 inference -------

matched to v2 but with a different matrix

------- v4: stiller scales -------

* Making sure to record distractor choices 

This was a mistake in the original--I didn't log whether you chose the null item or the logically true distractor in the salience condition (so I couldn't tell what proportion chose the null in the salience). 

* Make sure the prior is over the event being communicated about

Now Bob says "my favorite X has a Y" but you have empirical information about his long-range choice behavior.  In principle if this experiment looks good, this property should allow us to run a second version o

* Making sure they know the feature distribution in the test display

I've put back the "common grounding" manipulation check that asks how many features of each type there are. This is probably the biggest change. 

* More items to ensure generality

We now have snowmen, friends, christmas trees, bikes, sundaes, and pizzas. 

* Some small "base" changes to make sure that participants see different feature arrangements as reflecting three different types. 

This is pretty minor but helps to confirm that there's some variability in the object. So e.g. there's not just one friend, their faces vary on feature arrangement, color, and size, though it's pretty subtle. This actually doesn't matter much for christmas trees where you assume it's a different one each time, but my intuition is that it matters a lot for the "friends" item (because it's not just the same guy wearing different things).

* Fewer frequency conditions but more data in each condition

I ran 7 frequency conditions in the previous iteration--this time I'm doing 4: p(target) = [.125,.375,.625,.875]. Last time I had 50 in each condition (for a total of 700 participants = 50 x 7 x listener/salience). This time I will up it to 75 x 4 x 2 = 600. Relatively minor.

------- v5: stiller no scales mod -------

------- v6-7: getting base rate data with no schelling game  -------

just ask "which one is his friend" with no reference game or mumbles.

** note: there was a typo in experiments 4 and 5 such that the salience condition says "friend", e.g. for christmas tree, it would say "the friend I most like to trim has (mumble mumble)." this is not good. 


------- c1: Clean code -------

pragmods_c1.html is cleaned by adding comments and proper indentation.
The js file is partitiones into three js files with distinct functionality. 
Each of the .js file is polished and refactored to facilitate future changes
and improvements. Notice that the order in which the 
scripts are declared do matter a lot because they build on top of each other:

    <!-- The js scripts delaration section. To improve functionality there are a few js scripts so that parameters can be more easily set and functions be edited. The first one is pragmods_helper_functions_c1.js which contains simple functions. Note that these .js need to be declared in a proper order. -->
    <script src="pragmods_helper_functions_c1.js"></script>

    <!-- This .js file will set the variable random parameters of the experiment. When it comes to vectorize the parameters of the experiment using this -->
    <script src="pragmods_parameter_setter_c1.js"></script>

    <!-- This .js file is in charge of controlling the dynamic glow of the activities including constructing the text that is used in the familiarization task and -->
    <script src="pragmods_control_flow_c1.js"></script>

A general framework to write code that dynamically generates experiments from a state-space
of relevant possible experiments is to be found in pragmods_parameter_setter_c1.js


Bug with Fire Fox and Intermet Explorer:
It turns out that the end function had a *formatting* bug that Firefox and Internet
Explorer were not properly reading. However, beause this was compartamentalized in the code 
of pragmods_c1 and not before, the people with Firefox were probable unable to even start
the HIT in previous versions.


And bug with Interent Explorer:
getElementByName does not work as it does in every other browser... this in turn makes 
the check for the last button to fail no matter what. 
