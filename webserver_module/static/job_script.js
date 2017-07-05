var maxWidthMobile = 700;
var form_values = {};
form_values["skills"] = [];

$(document).ready(function() {
	adjustToSize();
	$( "#Datepicker" ).datepicker({ dateFormat: 'mm-dd-yy' });
	jobTypeSwitch();

    skill_input();

	$(window).resize(function() {
		adjustToSize();
	});

	$('#ReviewButton').bind('click', function(e) {
        $("#Status").text("Reviewing job ...").removeClass("Warning").removeClass("OK");
        $('#ReviewButton').prop("disabled",true).addClass("Disabled");

        $.getJSON($SCRIPT_ROOT + '/get_realtime_predictions', form_values).done(function(data) {
            $('#ReviewButton').prop("disabled",false).removeClass("Disabled");
            $("#Status").text("Job review complete").addClass("OK").removeClass("Warning");
        }).fail(function( jqxhr, textStatus, error ) {
            $('#ReviewButton').prop("disabled",false).removeClass("Disabled");
            $("#Status").text("Review failed").addClass("Warning").removeClass("OK");
        });
        e.preventDefault();
        return false;
    });

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
    var ind = form_values["skills"].indexOf(sel_skill);
    if (ind == -1) {
        form_values["skills"].push(sel_skill);

        var id_string = 'Token_' + sel_skill;
        $("#SkillsList").append("<span id='" + id_string + "' class='Token'>" + sel_skill + "</span>");
        input.value = "";
        $("#NoSkills").hide();

        $(("#" + id_string)).click(function () {
            $(this).remove();
            var index = form_values["skills"].indexOf(sel_skill);
            if (index > -1) {
                form_values["skills"].splice(index, 1);

                if (form_values["skills"].length == 0){
                    $("#NoSkills").show();
                }
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

function onValueInput(key, value){
    if (key != undefined && value != undefined && form_values[key] != value){
        form_values[key] = value

        // get predictions
        $("#Status").text("Updating recommendations ...").removeClass("Warning").removeClass("OK");
        $.getJSON($SCRIPT_ROOT + '/get_realtime_predictions', form_values).done(function(data) {
            // alert(data.result);
            var time = new Date();
            $("#Status").text("Recommendations updated at " + time.getHours() + ":" + time.getMinutes() + ":" + time.getSeconds()).addClass("OK").removeClass("Warning");
        }).fail(function( jqxhr, textStatus, error ) {
            $("#Status").text("Updating recommendations failed").addClass("Warning").removeClass("OK");
        });
    }
}