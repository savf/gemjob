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
    Highcharts.setOptions({
        colors: [ '#70ad47', 'green', '#c5e0b4', '#e2f0d9','#70ad47']
    });

	adjustToSize();
	$( "#Datepicker" ).datepicker({ dateFormat: 'mm-dd-yy' });

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

    jobTypeSwitch();

    $('#ExampleTextButton').bind('click', function(e) {
        getExampleText();
        e.preventDefault();
        return false;
    });

	$('#ModelButton').bind('click', function(e) {
        $("#Status").text("Loading predictions ...").removeClass("Warning").removeClass("OK");
        $('#ModelButton').prop("disabled",true).addClass("Disabled");

        $.getJSON($SCRIPT_ROOT + '/get_model_predictions', form_values).done(function (data) {
            if (data && data.result && Object.keys(data.result).length > 0) {
                var info = "<p style='margin-bottom:20px'>We use machine learning models to predict key metrics for the job-posting you prepared.</p>" +
                    "<p style='margin-bottom:20px'>In contrast to the real-time predictions, we use individual models for each metric which have " +
                    "all been precisely tuned to achieve more accurate results. These models do however require that all form values are filled, " +
                    "hence why they are only available as soon as the form has been filled in completely.</p>" +
                    "<br/><p style='margin-bottom:20px'>Click on any prediction to see which form values influenced the prediction the most.</p>";
                $("#PopUpContent").html("");
                $("#PopUpContent").append(info);
                var accordion_div = $("<div/>",{
                    id: "PopUpPredictions"
                });
                $("#PopUpContent").append(accordion_div);
                for(var key in data.result) {
                    if(key == 'budget') {
                        markupStats(key, "We propose a budget of $" + data.result[key].prediction.toFixed(0), data.result[key].stats, accordion_div)

                        if('KNN' in data.result) {
                            knnPrediction("budget", data.result['KNN'], accordion_div)
                        }
                    }
                    else if(key == 'feedback_for_client') {
                        markupStats(key, "We predict that freelancers working on this job will give you an overall feedback of " + data.result[key].prediction.toFixed(1), data.result[key].stats, accordion_div)
                    }
                    else if(key == 'job_type') {
                        prediction = data.result[key].prediction
                        if(prediction == "Hourly") {
                            markupStats(key, "This job might work best with hourly billing.", data.result[key].stats, accordion_div)
                        }
                        else {
                            markupStats(key, "We propose you make this a fixed-price job.", data.result[key].stats, accordion_div)
                        }

                    }

                }
                showPopUp("Model Predictions");
                $("#PopUpPredictions").accordion({
                    active: false,
                    collapsible: true,
                    activate: function( event, ui ) {
                        if(ui.newHeader.length > 0) {
                            adjustPopUp();
                        }
                    }
                });
                $("#Status").text("Model predictions complete").addClass("OK").removeClass("Warning");
                adjustPopUp()
            }
            else{
                $("#Status").text("Model predictions failed").addClass("Warning").removeClass("OK");
            }
            $('#ModelButton').prop("disabled",false).removeClass("Disabled");

        }).fail(function( jqxhr, textStatus, error ) {
            $('#ModelButton').prop("disabled",false).removeClass("Disabled");
            $("#Status").text("Model predictions failed").addClass("Warning").removeClass("OK");
        });
        e.preventDefault();
        return false;
    });

    $('#SubmitButton').bind('click', function(e) {
        if(is_client) {
            $("#Status").text("Submitting job ...").removeClass("Warning").removeClass("OK");
            $('#SubmitButton').prop("disabled", true).addClass("Disabled");

            $.getJSON($SCRIPT_ROOT + '/submit_job', form_values).done(function (data) {
                if (data && data.result && data.result.length > 0) {
                    $("#Status").text("Job submitted").addClass("OK").removeClass("Warning");
                    $("#FormSectionText").html("<a href='/' class='JobSent'>Success</a>")
                }
                else {
                    $("#Status").text("Submitting job failed").addClass("Warning").removeClass("OK");
                    showPopUp("Error", "Submitting job failed")
                }
                $('#SubmitButton').prop("disabled", false).removeClass("Disabled");

            }).fail(function (jqxhr, textStatus, error) {
                $('#SubmitButton').prop("disabled", false).removeClass("Disabled");
                $("#Status").text("Submitting job failed").addClass("Warning").removeClass("OK");
                showPopUp("Error", "Submitting job failed")
            });
        }
        else{
            showPopUp("Error", "You are logged in as a freelancer, the jobs created cannot be published.")
        }
        e.preventDefault();
        return false;
    });

    $('#UpdateButton').bind('click', function(e) {
        $("#Status").text("Updating job ...").removeClass("Warning").removeClass("OK");
        $('#UpdateButton').prop("disabled",true).addClass("Disabled");

        $.getJSON($SCRIPT_ROOT + '/update_job/id=' + job_id, form_values).done(function (data) {
            if (data && data.result && data.result == "updated") {
                $("#Status").text("Job updated").addClass("OK").removeClass("Warning");
                $("#FormSectionText").html("<a href='/' class='JobSent'>Success</a>")
            }
            else {
                $("#Status").text("Updating job failed").addClass("Warning").removeClass("OK");
                showPopUp("Error", "Updating job failed")
            }
            $('#UpdateButton').prop("disabled",false).removeClass("Disabled");

        }).fail(function( jqxhr, textStatus, error ) {
            $('#UpdateButton').prop("disabled",false).removeClass("Disabled");
            $("#Status").text("Updating job failed").addClass("Warning").removeClass("OK");
            showPopUp("Error", "Updating job failed")
        });

        e.preventDefault();
        return false;
    });

	$('#ModelButton').prop("disabled",true).addClass("Disabled");
	$('#SubmitButton').prop("disabled",true).addClass("Disabled");
	$('#UpdateButton').prop("disabled",true).addClass("Disabled");

	initWordCount();

});

function markupStats(attribute, title, stats, element) {
    var header = $("<h3>");
    header.html(title);
    // First the text-related attributes
    var stats_content = $("<div>");
    if('text' in stats) {
        var text_content = "<p>The form's text fields determine " + stats.text.importance.toFixed(0) + "% of the prediction, " +
            "where the title determines " + stats.title.importance.toFixed(0) +
            "%, the length of the title " + stats.title_length.importance.toFixed(0) +
            "%, the description " + stats.snippet.importance.toFixed(0) + "%, the " +
            "individual skill tags " + stats.skills.importance.toFixed(0) + "% and the " +
            "number of skills " + stats.skills_number.importance.toFixed(0) + "% of the overall prediction result.";
        delete stats.text; delete stats.title; delete stats.title_length;
        delete stats.snippet; delete stats.skills; delete stats.skills_number;
    }
    var normal_content = "<p> The non-text attributes influence the prediction with the following percentages:</p>";
    var bar_chart = $("<div/>",{
        id: "stats_chart_" + attribute,
        style: "min-width: 310px; max-width: 800px; height: 400px; margin: 0 auto"
    });
    var sorted_attributes = Object.keys(stats).sort(function(a,b){return stats[b].importance-stats[a].importance});
    var importances = [];
    var errors = [];
    for(var key in sorted_attributes) {
        importances.push(stats[sorted_attributes[key]].importance);
        errors.push(stats[sorted_attributes[key]].error);
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
        text: 'Importance of the non-text values'
    },
    xAxis: [{
        categories: sorted_attributes
    }],
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
        shared: true
    },

    series: [{
        showInLegend: false,
        name: 'Importance',
        type: 'column',
        data: importances,
        tooltip: {
            pointFormat: '<span style="font-weight: bold; color: {series.color}">{series.name}</span>: <b>{point.y:.2f} %</b> '
        }
    }, {
        type: 'errorbar',
        data: errors,
        tooltip: {
            pointFormat: '(error range: {point.low}-{point.high} %)<br/>'
        }
    }
    ]
});
}

function knnPrediction(attribute, data, element) {
    var median = false;
    try{
        data = JSON.parse(data);
        median = data[attribute+"_prediction"];
    } catch (err){
        median = false;
    }

    var title = "";
    var content = "";
    var chart = false;
    if (median === false) {
        title = "No second opinion for "+attribute+" found";
        content = "<p>We looked at 15 similar jobs to yours and all of them are missing this field!</p>";
    }
    else {
        title = "Second opinion on budget based on nearest neighbors: $"+median;
        content = "<p>This alternative recommendation is based on the median of the 15 most similar jobs to yours. As opposed to the real-time predictions, we used kNN instead of clustering. With all form fields filled, this approach has proven to be the most accurate even though less scalable than clustering.</p>" +
            "<p>Below, we show the statistics of the " + attribute + " attribute in a box plot:</p>";
        chart = $("<div/>", {
            id: "stats_chart_knn_" + attribute,
            style: "min-width: 310px; max-width: 800px; height: 400px; margin: 0 auto"
        });
    }

    var header = $("<h3>");
    header.html(title);
    var stats_content = $("<div>");

    element.append(header);
    stats_content.html(content);
    element.append(stats_content);

    if (chart !== false) {
        stats_content.append(chart);

        Highcharts.chart('stats_chart_knn_' + attribute, {
            chart: {
                type: 'boxplot'
            },
            title: {
                text: ""
            },
            legend: {
                enabled: false
            },
            yAxis: {
                title: {
                    text: element.attr("id")
                },
                plotLines: [{
                    value: data[attribute + "_mean"],
                    color: 'green',
                    width: 1,
                    zIndex: 20,
                    label: {
                        text: 'Mean: ' + data[attribute + "_mean"],
                        align: 'center',
                        style: {
                            color: 'darkgreen'
                        }
                    }
                }]
            },
            series: [{
                name: element.attr("id"),
                data: [
                    [data[attribute + "_min"],
                        data[attribute + "_25quantile"],
                        median,
                        data[attribute + "_75quantile"],
                        data[attribute + "_max"]]
                ]
            }]

        });
    }
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
        onValueInput("workload", $("#WorkloadSelect").val(), false);
    }
    else{
        $(".IfHourly").hide();
        $(".IfFixed").show();
        delete form_values["workload"];
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
            form_values["skills"] = getSkillsString();
        }
    });

    form_values["skills"] = getSkillsString();
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
    popUp.show()
    adjustPopUp()
}

function hidePopUp(){
    popUpBackground.hide();
    popUp.hide();
}

function adjustPopUp(){
    popUp.css("top", Math.max(0, (($(window).height() - popUp.outerHeight()) / 2) + $(window).scrollTop()) + "px");
}

function updateRealTimePredictions(){

    // get predictions
    $("#Status").text("Updating recommendations ...").removeClass("Warning").removeClass("OK");

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
                "<tr valign='top'> <td>Value counts:</td>  <td>" + cluster_predictions[element.attr("id") + "_value_counts"] + "</td> </tr> </table>" +
                "<div id='BoxPlotLiveRecommendation'></div>";

            showPopUp(element.attr("name") + ":", content);

            var value_counts_names = [];
            var value_counts_values = [];
            var value_counts = cluster_predictions[element.attr("id") + "_value_counts"].split("<br>");
            for (var i = 0; i < value_counts.length-1; i++){
                var name_value_pair = value_counts[i].split(":");
                value_counts_names.push(name_value_pair[0]);
                var v_begin = name_value_pair[2].indexOf("'>")+2;
                var v_end = name_value_pair[2].indexOf("</");
                value_counts_values.push(parseInt(name_value_pair[2].substring(v_begin, v_end)));
            }

            Highcharts.chart('BoxPlotLiveRecommendation', {
                chart: {
                    type: 'bar'
                },
                title: {
                    text: 'Value counts in cluster'
                },
                xAxis: {
                    categories: value_counts_names,
                    title: {
                        text: null
                    }
                },
                yAxis: {
                    min: 0,
                    title: {
                        text: 'Count',
                        align: 'high'
                    },
                    labels: {
                        overflow: 'justify'
                    }
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
                    data: value_counts_values
                }]
            });
            adjustPopUp()
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
                "<p>" + hint + "</p>"+
                "<div id='BoxPlotLiveRecommendation'></div>";

            showPopUp(element.attr("name") + ":", content);

            Highcharts.chart('BoxPlotLiveRecommendation', {
                chart: {
                    type: 'boxplot'
                },
                title: {
                    text: ""
                },
                legend: {
                    enabled: false
                },
                yAxis: {
                    title: {
                        text: element.attr("id")
                    },
                    plotLines: [{
                        value: cluster_predictions[element.attr("id") + "_mean"],
                        color: 'green',
                        width: 1,
                        zIndex: 20,
                        label: {
                            text: 'Mean: '+cluster_predictions[element.attr("id") + "_mean"],
                            align: 'center',
                            style: {
                                color: 'darkgreen'
                            }
                        }
                    }]
                },
                series: [{
                    name: element.attr("id"),
                    data: [
                        [   cluster_predictions[element.attr("id") + "_min"],
                            cluster_predictions[element.attr("id") + "_25quantile"],
                            median,
                            cluster_predictions[element.attr("id") + "_75quantile"],
                            cluster_predictions[element.attr("id") + "_max"]]
                    ]
                }]

            });
            adjustPopUp()
        }
    }
}

function initWordCount() {
    updateWordCount($("#TitleInput"), $("#TitleWordCount"));
    updateWordCount($("#SnippetTextArea"), $("#SnippetWordCount"));
    $("#TitleInput").keyup(function() {
        updateWordCount($("#TitleInput"), $("#TitleWordCount"));
    });
    $("#SnippetTextArea").keyup(function() {
        updateWordCount($("#SnippetTextArea"), $("#SnippetWordCount"));
    });
}

function updateWordCount(inputField, outputField) {
    var word_num = 0;
    if (inputField.val() != "")
        word_num = inputField.val().split(" ").length;
    var exp_string = " word";
    if (word_num != 1)
        exp_string+="s";
    outputField.text(word_num + exp_string);
}

function getExampleText(){

    // get predictions
    $("#Status").text("Getting sample texts ...").removeClass("Warning").removeClass("OK");

	// make sure text and title are updated, if button was pressed right after writing into text field
	form_values["title"] = $("#SnippetTextArea").val();
	form_values["snippet"] = $("#TitleInput").val();

    $.getJSON($SCRIPT_ROOT + '/get_knn_predictions/target=text', form_values).done(function (data) {
        var time = new Date();
        try{
            var knn_predictions = JSON.parse(data.result);

            var content = "<p>The title and description shown below are taken from the most similar job to yours.</p>" +
                    "<br><h2>Title:</h2>" +
                    "<p>"+ knn_predictions["title_prediction"] +"</p>" +
                    "<br><h2>Description:</h2>" +
                    "<p>"+ knn_predictions["snippet_prediction"] +"</p>";

			showPopUp("Example Text:", content);

            $("#Status").text("Sample texts loaded at " + time.getHours() + "h" + time.getMinutes() + "min" + time.getSeconds() + "s").addClass("OK").removeClass("Warning");
        }
        catch (err) {
            console.log("# Live Recommendation Error:");
            console.log(err);
            $("#Status").text("Getting sample texts failed at " + time.getHours() + "h" + time.getMinutes() + "min" + time.getSeconds() + "s").addClass("Warning").removeClass("OK");
        }
    }).fail(function (jqxhr, textStatus, error) {
        var time = new Date();
        $("#Status").text("Getting sample texts failed at " + time.getHours() + "h" + time.getMinutes() + "min" + time.getSeconds() + "s").addClass("Warning").removeClass("OK");
    });
}
