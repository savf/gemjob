var maxWidthMobile = 700;
var min_filled_for_predictions = 14;
var min_filled_for_models = 17;
var form_elements = {};
var recommendation_elements = {};
var recommendation_labels = {};
var cluster_predictions = 0;
var skills_selected = [];
form_values["skills"] = "";

$(document).ready(function() {
	adjustToSize();
	$( "#Datepicker" ).datepicker({ dateFormat: 'mm-dd-yy' });
	jobTypeSwitch();

    skill_input();

    initPopUp();

    form_elements = $(".ReadInput");
    for (var i = 0; i < form_elements.length; i++) {
        onValueInput(form_elements[i].name, form_elements[i].value, true);
    }

    recommendation_elements = $(".LiveRecommendation");
    recommendation_labels = $(".RecommendationLabel");
    recommendation_labels.hide();

    recommendation_elements.bind('click', function(e) {
        var clicked = $(this);
        showStats(clicked);
        e.preventDefault();
        return false;
    });

    $("#feedback_for_client").bind('click', function(e) {
        var clicked = $(this);
        showStats(clicked);
        e.preventDefault();
        return false;
    });

	$(window).resize(function() {
		adjustToSize();
	});

    for (var i = 0; i < skills_preselected.length; i++){
        addSkillToList(skills_preselected[i]);
    }

	$('#ModelButton').bind('click', function(e) {
        $("#Status").text("Loading predictions ...").removeClass("Warning").removeClass("OK");
        $('#ModelButton').prop("disabled",true).addClass("Disabled");

        // read all form fields
        for (var i = 0; i < form_elements.length; i++) {
            onValueInput(form_elements[i].name, form_elements[i].value, true);
        }

        $.getJSON($SCRIPT_ROOT + '/get_model_predictions', form_values).done(function (data) {
            if (data && data.result && Object.keys(data.result).length > 0) {
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
                $("#Status").text("Model predictions complete").addClass("OK").removeClass("Warning");
            }
            else{
                $("#Status").text("Model predictions failed").addClass("Warning").removeClass("OK");
            }
            $('#ModelButton').prop("disabled",false).removeClass("Disabled");

        }).fail(function( jqxhr, textStatus, error ) {
            $('#ModelButton').prop("disabled",false).removeClass("Disabled");
            $("#Status").text("Review failed").addClass("Warning").removeClass("OK");
        });
        e.preventDefault();
        return false;
    });
	$('#ModelButton').prop("disabled",true).addClass("Disabled");
	$('#SubmitButton').prop("disabled",true).addClass("Disabled");
	$('#UpdateButton').prop("disabled",true).addClass("Disabled");

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
        delete form_values["budget"];
        $("#BudgetInput").val("");
        min_filled_for_models = 17;
    }
    else{
        $(".IfHourly").hide();
        $(".IfFixed").show();
        min_filled_for_models = 18;
    }
}

function addSkill () {
    var input = document.getElementById("SkillSearch");
    var skillItem = input.value;
    input.value = "";
    var ind = skills_selected.indexOf(skillItem);
    if (ind == -1) {
        addSkillToList(skillItem);
        updateRealTimePredictions();
    }
}

function addSkillToList(skillItem){
    skills_selected.push(skillItem);

    var id_string = 'Token_' + skillItem.replace(/#|\,|\.|\"|\?|\!|\*/g,'_');
    $("#SkillsList").append("<span id='" + id_string + "' class='Token'>" + skillItem + "</span>");
    $("#NoSkills").hide();

    $(("#" + id_string)).click(function () {
        $(this).remove();
        var index = skills_selected.indexOf(skillItem);
        if (index > -1) {
            skills_selected.splice(index, 1);

            if (skills_selected.length == 0){
                $("#NoSkills").show();
            }

            updateRealTimePredictions();
        }
    });
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

            if(count < min_filled_for_models) {
                $('#ModelButton').prop("disabled", true).addClass("Disabled");
                $('#SubmitButton').prop("disabled", true).addClass("Disabled");
                $('#UpdateButton').prop("disabled",true).addClass("Disabled");
            }
            else {
                $('#ModelButton').prop("disabled", false).removeClass("Disabled");
                $('#SubmitButton').prop("disabled", false).removeClass("Disabled");
                $('#UpdateButton').prop("disabled",false).removeClass("Disabled");
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
    popUp.css("top", Math.max(0, (($(window).height() - popUp.outerHeight()) / 2) + $(window).scrollTop()) + "px");
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
            var init = false;
            if (cluster_predictions === 0) init = true;

            cluster_predictions = JSON.parse(data.result);

            recommendation_elements.each(function() {
                var current_el = $( this );
                var current_id = current_el.attr('id');
                var rec_value = cluster_predictions[current_id];
                if (rec_value == undefined)
                    rec_value = "undefined";
                current_el.text(rec_value);

                if(form_values[current_id] && current_id != "experience_level") {

                    if (typeof rec_value === 'string' && rec_value.toLowerCase() != form_values[current_id].toLowerCase() &&
                    !(rec_value == "Fixed" && form_values[current_id] == "fixed-price"))
                        current_el.addClass("DifferentPrediction");
                    else if (!(typeof rec_value === 'string') && rec_value != form_values[current_id])
                        current_el.addClass("DifferentPrediction");
                    else
                        current_el.removeClass("DifferentPrediction");

                }
                else if (current_id == "total_hours" && form_values["duration"]){
                    if (rec_value != form_values["duration"])
                        current_el.addClass("DifferentPrediction");
                    else
                        current_el.removeClass("DifferentPrediction");
                }
                else if (current_id == "experience_level"){
                    if(     (rec_value == "Entry Level" && form_values["experience_level"] == 1) ||
                            (rec_value == "Intermediate" && form_values["experience_level"] == 2) ||
                            (rec_value == "Expert" && form_values["experience_level"] == 3))
                        current_el.removeClass("DifferentPrediction")
                    else current_el.addClass("DifferentPrediction");
                }
                else current_el.removeClass("DifferentPrediction");
            });

            $("#cluster_size").text(cluster_predictions["cluster_size"]);
            $("#feedback_for_client").text(cluster_predictions["feedback_for_client"]);
            recommendation_labels.show();

            var text_field_names = ["title", "snippet"];
            for (var i = 0; i < text_field_names.length; i++) {
                if(form_values[text_field_names[i]]) {
                    var wordCount = form_values[text_field_names[i]].split(" ").length;
                    if (cluster_predictions[text_field_names[i] + "_length"] != wordCount)
                        $("#" + text_field_names[i] + "_length").addClass("DifferentPrediction");
                    else $("#" + text_field_names[i] + "_length").removeClass("DifferentPrediction");
                }
            }
            if (cluster_predictions["skills_number"] != skills_selected.length)
                $("#skills_number").addClass("DifferentPrediction");
            else $("#skills_number").removeClass("DifferentPrediction");

            if(init) $("#feedback_for_client").addClass("Clickable");

            $("#Status").text("Recommendations updated at " + time.getHours() + "h" + time.getMinutes() + "min" + time.getSeconds() + "s").addClass("OK").removeClass("Warning");
        }
        catch (err) {
            console.log("# Live Recommendation Error:");
            console.log(err);
            $("#Status").text("Updating recommendations failed at " + time.getHours() + "h" + time.getMinutes() + "min" + time.getSeconds() + "s").addClass("Warning").removeClass("OK");
        }
    }).fail(function (jqxhr, textStatus, error) {
        var time = new Date();
        $("#Status").text("Updating recommendations failed at " + time.getHours() + "h" + time.getMinutes() + "min" + time.getSeconds() + "s").addClass("Warning").removeClass("OK");
    });
}

function showStats(element){
    if (cluster_predictions != 0) {
        var content = "<p>We found " + cluster_predictions["cluster_size"] + " similar jobs to yours based on mean-shift clustering.</p>" +
            "<br><h2>Stats:</h2>" +
            "<table class='BreakDownTable'> <tr valign='top'> <td>Attribute name:</td> <td>" + element.attr("id") + "</td> </tr> ";

        if (typeof cluster_predictions[element.attr("id")] === 'string') {
            content += "<tr valign='top'> <td>Majority:</td>  <td>" + cluster_predictions[element.attr("id")] + "</td> </tr>" +
                "<tr valign='top'> <td>Value counts:</td>  <td>" + cluster_predictions[element.attr("id") + "_value_counts"] + "</td> </tr> </table>";
        }
        else {
            var median = cluster_predictions[element.attr("id")];
            var hint = "";
            if (median == undefined) {
                median = "undefined";
                hint = "All jobs in the cluster are missing this field!"
            }
            content += "<tr valign='top'> <td>Mean:</td>  <td>" + cluster_predictions[element.attr("id") + "_mean"] + "</td> </tr>" +
                "<tr valign='top'> <td>Std:</td>  <td>" + cluster_predictions[element.attr("id") + "_std"] + "</td> </tr> " +
                "<tr valign='top'> <td>Min:</td>  <td>" + cluster_predictions[element.attr("id") + "_min"] + "</td> </tr> " +
                "<tr valign='top'> <td>1st quantile:</td>  <td>" + cluster_predictions[element.attr("id") + "_25quantile"] + "</td> </tr> " +
                    "<tr valign='top'> <td>Median:</td>  <td>" + median + "</td> </tr> " +
                "<tr valign='top'> <td>3rd quantile:</td>  <td>" + cluster_predictions[element.attr("id") + "_75quantile"] + "</td> </tr> " +
                "<tr valign='top'> <td>Max:</td>  <td>" + cluster_predictions[element.attr("id") + "_max"] + "</td> </tr> " +
                "</table>" +
                "<p>" + hint + "</p>";
        }


        showPopUp(element.attr("name") + ":", content);
    }
}