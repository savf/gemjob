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
                var info = "<p style='margin-bottom:20px'>We use machine learning models to predict key metrics for the job-posting you prepared.</p>" +
                    "<p style='margin-bottom:20px'>In contrast to the real-time predictions, we use individual models for each metric which have " +
                    "all been precisely tuned to achieve very accurate results. These models do however require that all form values are filled, " +
                    "hence why they are only available as soon as the form has been filled in completely.</p>" +
                    "<br/><p style='margin-bottom:20px'>Click on any prediction to see which form values influenced the prediction the most.</p>";
                $("#PopUpContent").html("")
                $("#PopUpContent").append(info)
                var accordion_div = $("<div/>",{
                    id: "PopUpPredictions"
                });
                $("#PopUpContent").append(accordion_div);
                for(var key in data.result) {
                    if(key == 'budget') {
                        markupStats(key, "We propose a budget of " + data.result[key].prediction.toFixed(0), data.result[key].stats, accordion_div)
                    }
                    else if(key == 'feedback_for_client') {
                        markupStats(key, "We predict that freelancers working on this job will give you an overall feedback of " + data.result[key].prediction.toFixed(1), data.result[key].stats, accordion_div)
                    }

                }
                showPopUp("Model Predictions");
                $("#PopUpPredictions").accordion({
                    active: false,
                    collapsible: true
                });
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

function markupStats(attribute, title, stats, element) {
    var header = $("<h3>");
    header.html(title);
    // First the text-related attributes
    var stats_content = $("<div>");
    if('text' in stats) {
        var text_content = "<p>The form's text fields determine " + (stats.text*100).toFixed(0) + "% of the prediction, " +
            "where the title determines " + (stats.title*100).toFixed(0) +
            "%, the length of the title " + (stats.title_length*100).toFixed(0) +
            "%, the description " + (stats.snippet*100).toFixed(0) + "%, the " +
            "individual skill tags " + (stats.skills*100).toFixed(0) + "% and the " +
            "number of skills " + (stats.skills_number*100).toFixed(0) + "% of the overall prediction result.";
        delete stats.text; delete stats.title; delete stats.title_length;
        delete stats.snippet; delete stats.skills; delete stats.skills_number;
    }
    var normal_content = "<p> The non-text attributes influence the prediction with the following percentages:</p>";
    var bar_chart = $("<div/>",{
        id: "stats_chart_" + attribute,
        style: "min-width: 310px; max-width: 800px; height: 400px; margin: 0 auto"
    });
    var sorted_attributes = Object.keys(stats).sort(function(a,b){return stats[b]-stats[a]});
    var importances = [];
    for(var key in sorted_attributes) {
        importances.push(Math.round((stats[sorted_attributes[key]]*100) * 1e2) / 1e2)
    }
    for(var key in sorted_attributes) {
        sorted_attributes[key] = dict[sorted_attributes[key]].en
    }
    element.append(header);
    stats_content.html(text_content + normal_content);
    element.append(stats_content);
    stats_content.append(bar_chart);
    // Bar chart container has been added to the DOM -> generate bar chart
    Highcharts.chart('stats_chart_' + attribute, {
        chart: {
            type: 'bar'
        },
        title: {
            text: 'Importance of the non-text Values'
        },
        xAxis: {
            categories: sorted_attributes,
            title: {
                text: null
            }
        },
        yAxis: {
            min: 0,
            title: {
                text: 'Importance (percentage)',
                align: 'high'
            },
            labels: {
                overflow: 'justify'
            }
        },
        tooltip: {
            valueSuffix: ' %'
        },
        plotOptions: {
            bar: {
                dataLabels: {
                    enabled: true
                }
            }
        },
        credits: {
            enabled: true
        },
        series: [{
            showInLegend: false,
            data: importances
        }]
    });
}

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
    if(htmlContent !== undefined) {
        popUpContent.html(htmlContent);
    }
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
