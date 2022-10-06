from flask_wtf import FlaskForm
from wtforms import MultipleFileField, FieldList
from wtforms.validators import InputRequired
from wtforms import validators
from wtforms.fields import SubmitField, SelectField
from collections.abc import Iterable
from werkzeug.datastructures import FileStorage
from wtforms.validators import StopValidation

# Needed for validationg multiple image files at once
class MultiFileAllowed(object):
    def __init__(self, upload_set, message=None):
        self.upload_set = upload_set
        self.message = message

    def __call__(self, form, field):

        if not (
            all(isinstance(item, FileStorage) for item in field.data) and field.data
        ):
            return

        for data in field.data:
            filename = data.filename.lower()

            if isinstance(self.upload_set, Iterable):
                if any(filename.endswith("." + x) for x in self.upload_set):
                    return

                raise StopValidation(
                    self.message
                    or field.gettext(
                        "File does not have an approved extension: {extensions}"
                    ).format(extensions=", ".join(self.upload_set))
                )

            if not self.upload_set.file_allowed(field.data, filename):
                raise StopValidation(
                    self.message
                    or field.gettext("File does not have an approved extension.")
                )


# Create image form
class PhotoForm(FlaskForm):
    photos = MultipleFileField(
        "Images",
        validators=[
            InputRequired(),
            MultiFileAllowed(["png", "webp", "jpg", "jpeg"], "Only images allowed!"),
        ],
    )
    submit = SubmitField("Upload")


# Create label form
class LabelForm(FlaskForm):
    labels = FieldList(
        SelectField(
            "Choose Label",
            choices=[("Choose label..."), ("scratch"), ("dent"), ("rim"), ("other")],
        ),
        min_entries=50,
    )
    submit = SubmitField("Export Labels")


# Create correction form
class CorrectForm(FlaskForm):
    labels = FieldList(
        SelectField(
            "Choose Label",
            choices=[("Choose label..."), ("scratch"), ("dent"), ("rim"), ("other")],
        ),
        [validators.DataRequired()],
        min_entries=50,
    )
    submit = SubmitField("Retrain Model")
