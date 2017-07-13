var maxWidthMobile = 700;
var min_filled_for_predictions = 11;
skills_selected = [];
var form_elements = {};
var recommendation_elements = {};

$(document).ready(function() {
	adjustToSize();
	$( "#Datepicker" ).datepicker({ dateFormat: 'mm-dd-yy' });
	jobTypeSwitch();

    skill_input();

    form_elements = $(".ReadInput");
    for (var i = 0; i < form_elements.length; i++) {
        onValueInput(form_elements[i].name, form_elements[i].value, true);
    }

    recommendation_elements = $(".LiveRecommendation");

	$(window).resize(function() {
		adjustToSize();
	});

	initPopUp();

	$('#ReviewButton').bind('click', function(e) {
        $("#Status").text("Reviewing job ...").removeClass("Warning").removeClass("OK");
        $('#ReviewButton').prop("disabled",true).addClass("Disabled");

        // read all form fields
        for (var i = 0; i < form_elements.length; i++) {
            onValueInput(form_elements[i].name, form_elements[i].value, true);
        }

        $.getJSON($SCRIPT_ROOT + '/get_model_predictions', form_values).done(function (data) {
            if (data && data.result) {
                var content = '';
                for(var key in data.result) {
                    if(key == 'budget') {
                        content += 'We propose a budget of ' + data.result[key] + '<br>';
                    }
                    else if(key == 'client_feedback') {
                        content += 'Your overall feedback might change to ' + data.result[key] + '<br>';
                    }
                    else if(key == 'job_type') {
                        content += 'You should make this a ' + data.result[key] + ' job </br>';
                    }
                }
                showPopUp("Model Predictions", content);
            }
            $('#ReviewButton').prop("disabled",false).removeClass("Disabled");
            $("#Status").text("Job review complete").addClass("OK").removeClass("Warning");
        }).fail(function( jqxhr, textStatus, error ) {
            $('#ReviewButton').prop("disabled",false).removeClass("Disabled");
            $("#Status").text("Review failed").addClass("Warning").removeClass("OK");
        });
        e.preventDefault();
        return false;
    });
	$('#ReviewButton').prop("disabled",true).addClass("Disabled");
	$('#SubmitButton').prop("disabled",true).addClass("Disabled");

});

function adjustToSize() {
    var isMobileSize = $(window).width() <= maxWidthMobile;

    if (isMobileSize){
        $(".Label").css({"max-width": "100%", "min-width": "100%"});
        $(".LabeledElement").css({"max-width": "100%", "min-width": "100%", "margin-left": "0%"});
    }
    else {
        $(".Label").css({"max-width": "25%", "min-width": "25%"});
        $(".LabeledElement").css({"max-width": "70%", "min-width": "70%", "margin-left": "5%"});
    }
}

function jobTypeSwitch() {
    if (document.getElementById("JobType").value == "hourly"){
        $(".IfHourly").show();
        $(".IfFixed").hide();
    }
    else{
        $(".IfHourly").hide();
        $(".IfFixed").show();
    }
}

function addSkill () {
    var input = document.getElementById("SkillSearch");
    var sel_skill = input.value;
    var ind = skills_selected.indexOf(sel_skill);
    if (ind == -1) {
        skills_selected.push(sel_skill);

        var id_string = 'Token_' + sel_skill.replace(/#|\,|\.|\"|\?|\!|\*/g,'_');
        $("#SkillsList").append("<span id='" + id_string + "' class='Token'>" + sel_skill + "</span>");
        input.value = "";
        $("#NoSkills").hide();

        updateRealTimePredictions();

        $(("#" + id_string)).click(function () {
            $(this).remove();
            var index = skills_selected.indexOf(sel_skill);
            if (index > -1) {
                skills_selected.splice(index, 1);

                if (skills_selected.length == 0){
                    $("#NoSkills").show();
                }

                updateRealTimePredictions();
            }
        });
    }
}

function skill_input() {
    var input = document.getElementById("SkillSearch");
    new Awesomplete(input, {
        list: skills_list,
        minChars: 1
    });

    input.addEventListener("awesomplete-selectcomplete", addSkill, false);
}

function getSkillsString(){
    var skills_selected_string = "";
    for (var i = 0; i < skills_selected.length; i++) {
        skills_selected_string = skills_selected_string + " " + skills_selected[i];
    }
    return skills_selected_string;
}

function onValueInput(key, value, doNotPredict){
    if (key != undefined && form_values[key] != value){
        if(value == "" || value == undefined)
            delete form_values[key];
        else
            form_values[key] = value;

        if (!doNotPredict) {
            var count = Object.keys(form_values).length;

            if(count < 16) {
                $('#ReviewButton').prop("disabled", true).addClass("Disabled");
                $('#SubmitButton').prop("disabled", true).addClass("Disabled");
            }
            else {
                $('#ReviewButton').prop("disabled", false).removeClass("Disabled");
                $('#SubmitButton').prop("disabled", false).removeClass("Disabled");
            }

            if (count > min_filled_for_predictions) {
                updateRealTimePredictions();
            }
        }
    }
}


function initPopUp(){
    popUpBackground = $("#PopUpBackground");
    popUp = $("#PopUp");
    popUpTitle = $("#PopUpTitle");
    popUpContent = $("#PopUpContent");
    popUpBackground.bind('click', function(e) {
        hidePopUp();
    });
}

function showPopUp(title, htmlContent) {
    popUpTitle.text(title);
    popUpContent.html(htmlContent);
    popUpBackground.show();
    popUp.show();
}

function hidePopUp(){
    popUpBackground.hide();
    popUp.hide();
}

function updateRealTimePredictions(){

    // get predictions
    $("#Status").text("Updating recommendations ...").removeClass("Warning").removeClass("OK");
    form_values["skills"] = getSkillsString();

    $.getJSON($SCRIPT_ROOT + '/get_realtime_predictions', form_values).done(function (data) {
        var time = new Date();
        try{
            // showPopUp("Cluster Predictions", data.result);
            cluster_predictions = JSON.parse(data.result);

            recommendation_elements.each(function() {
                var current_el = $( this );
                var rec_value = cluster_predictions[current_el.attr('id')];
                if (rec_value == -1)
                    rec_value = "None";
                current_el.text(rec_value);

                if(form_values[current_el.attr('id')] && current_el.attr('id') != "experience_level") {
                    if (typeof rec_value === 'string' && rec_value.toLowerCase() != form_values[current_el.attr('id')].toLowerCase() &&
                    !(rec_value == "Fixed" && form_values[current_el.attr('id')] == "fixed-price"))
                        current_el.addClass("DifferentPrediction");
                    else if (!(typeof rec_value === 'string') && rec_value != form_values[current_el.attr('id')])
                        current_el.addClass("DifferentPrediction");
                    else
                        current_el.removeClass("DifferentPrediction");
                }
                else if (current_el.attr('id') == "total_hours" && form_values["duration"]){
                    if (rec_value != form_values["duration"])
                        current_el.addClass("DifferentPrediction");
                    else
                        current_el.removeClass("DifferentPrediction");
                }
                else if (current_el.attr('id') == "experience_level"){
                    if(     (rec_value == "Entry Level" && form_values["experience_level"] == 1) ||
                            (rec_value == "Intermediate" && form_values["experience_level"] == 2) ||
                            (rec_value == "Expert" && form_values["experience_level"] == 3))
                        current_el.removeClass("DifferentPrediction")
                    else current_el.addClass("DifferentPrediction");
                }
                else current_el.removeClass("DifferentPrediction");
            });
            $("#cluster_size").text(cluster_predictions["cluster_size"]);

            $("#Status").text("Recommendations updated at " + time.getHours() + "h" + time.getMinutes() + "min" + time.getSeconds() + "s").addClass("OK").removeClass("Warning");
        }
        catch (err) {
            $("#Status").text("Updating recommendations failed at " + time.getHours() + "h" + time.getMinutes() + "min" + time.getSeconds() + "s").addClass("Warning").removeClass("OK");
        }
    }).fail(function (jqxhr, textStatus, error) {
        var time = new Date();
        $("#Status").text("Updating recommendations failed at " + time.getHours() + "h" + time.getMinutes() + "min" + time.getSeconds() + "s").addClass("Warning").removeClass("OK");
    });
}