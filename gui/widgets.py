# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 17:49:56 2016

@author: hannes
"""
import inspect
from importlib import import_module
import sys
import guidata.qt.QtGui as qtgui
import guidata.qt.QtCore as QtCore
from guidata.configtools import get_icon
from guidata.qt.QtGui import QApplication, QWidget, \
    QVBoxLayout, QListWidget, QAbstractItemView, \
    QPushButton, QComboBox, QListWidgetItem, \
    QLineEdit, QLabel, QSpinBox, QCheckBox, QHBoxLayout, \
    QToolBar, QGridLayout, QDialog, QDialogButtonBox
import re

import logging as l
l.basicConfig(level=l.DEBUG)


class QSliceInput(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.layout = QHBoxLayout()
        self.widgets = []
        for i, label_text in enumerate(["start", "stop", "step"]):
            label = QLabel(label_text)
            argument_widget = QLineEdit()

            self.layout.addWidget(label)
            self.layout.addWidget(argument_widget)
            self.widgets.append(argument_widget)
        self.setLayout(self.layout)
        self.slice = slice(None)
        
    def setSlice(self, slice):
        if slice is None:
            return False
        self.widgets[0].setText(str(slice.start))
        self.widgets[1].setText(str(slice.stop))
        self.widgets[2].setText(str(slice.step))
    
    def getSlice(self):
        return slice(*[int(w.text()) for w in self.widgets])
        
    
class OperationsWidget(QWidget):
    operations_changed = QtCore.pyqtSignal()
    operation_changed = QtCore.pyqtSignal(dict, int)
    operation_added = QtCore.pyqtSignal(dict, int)

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        self.widget_layout = QVBoxLayout()

        title_layout = QHBoxLayout()
        title_layout.addStretch()
        style = "<span style=\'color: #444444\'><b>%s</b></span>"
        title = QLabel(style % "Operations")
        title_layout.addWidget(title)
        title_layout.addStretch()
        self.widget_layout.addLayout(title_layout)

        # Create ListWidget and add 10 items to move around.
        self.list_widget = QListWidget()

        # self.list_widget.setDragDropMode(QAbstractItemView.InternalMove)
        self.list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_widget.setSortingEnabled(False)
        self.list_widget.currentItemChanged.connect(self._populate_settings_update)
        self.widget_layout.addWidget(self.list_widget)

        otitle_layout = QHBoxLayout()
        otitle_layout.addStretch()
        otitle = QLabel(style % "Operation settings")
        otitle_layout.addWidget(otitle)
        otitle_layout.addStretch()
        self.widget_layout.addLayout(otitle_layout)

        self.operations_combo = QComboBox()
        self.operations_combo.currentIndexChanged.connect(self._populate_settings_add)
        self.widget_layout.addWidget(self.operations_combo)

        self.operation_settings = GenericOperationWidget()
        self.widget_layout.addWidget(self.operation_settings)

        self.toolbar = QToolBar()
        self.toolbar.addAction(get_icon('apply.png'), "Apply/Replace",
                               self._change_operation)
        self.toolbar.addAction(get_icon('editors/edit_add.png'), "Add after",
                               self._add_operation)
        self.toolbar.addAction(get_icon('trash.png'), "Remove",
                               self._remove_operation)


        self.widget_layout.addWidget(self.toolbar)
        self.setLayout(self.widget_layout)

    def populate_available_operations(self, dict):
        """
        Populate combobox with available operation names
        """
        self.operations_combo.addItems(dict)

    def set_operations(self, operations_dict):
        """
        Populate operations list with given dict of operations
        """
        self.list_widget.clear()
        for op in operations_dict:
            self.list_widget.addItem(Operation(op))

    def get_operations(self):
        """
        Return list of operations.
        """
        operations = []
        for i in range(self.list_widget.count()):
            op = self.list_widget.item(i)
            operations.append(op._op)

        return operations

    def _remove_operation(self):
        self.list_widget.takeItem(self.list_widget.currentRow())
        self.operations_changed.emit()

    def _add_operation(self):
        """
        Add operation currently in self.operation_settings to the operation
        list.

        Signals:
        ========
        Emits self.opeartion_added and self.operations_changed on success
        """
        op = self.operation_settings.get_operation()
        current_row = self.list_widget.currentRow()
        self.list_widget.insertItem(current_row + 1, Operation(op))
        index = self.list_widget.model().index(current_row + 1, 0)
        self.list_widget.setCurrentIndex(index)
        self.operation_added.emit(op, current_row + 1)
        self.operations_changed.emit()

    def _change_operation(self):
        """
        Replace currently selected operation with operation in
        self.operation_settings (apply changed settings or replace operation).

        Signals:
        ========
        Emits self.operation_changed and self.operations_changed on success
        """
        op = self.operation_settings.get_operation()
        current_row = self.list_widget.currentRow()
        self.list_widget.takeItem(self.list_widget.currentRow())
        self.list_widget.insertItem(current_row, Operation(op))
        index = self.list_widget.model().index(current_row, 0)
        self.list_widget.setCurrentIndex(index)

        self.operation_changed.emit(op, current_row)
        self.operations_changed.emit()


    def _populate_settings_update(self, item):
        """
        Fill self.operation_settings with details of currently selected
        operation.
        """
        try:
            idx = self.operations_combo.findText(item._op["module"])
            if idx >= 0:
                self.operations_combo.setCurrentIndex(idx)
            self.operation_settings.set_operation(item._op)
        except AttributeError:
            pass

    def _populate_settings_add(self, index):
        self.operation_settings.set_operation({"module": self.operations_combo.currentText()})



class GenericOperationWidget(QWidget):
    type_mapping = {"int": {"widget": QSpinBox,
                            "display_func": "setValue",
                            "display_conversion": int,
                            "get_func": "value",
                            "get_conversion": int},
                    "float": {"widget": QLineEdit,
                              "display_func": "setText",
                              "display_conversion": str,
                              "get_func": "text",
                              "get_conversion": float},
                    "bool": {"widget": QCheckBox,
                             "display_func": "setChecked",
                             "get_func": "checkState"},
                    "slice": {"widget": QSliceInput,
                              "display_func": "setSlice",
                              "get_func": "getSlice"}}
    def __init__(self, parent=None, op=None):
        QWidget.__init__(self, parent)
        self.widget_layout = QVBoxLayout()
        self.minimumHeight = 200 # FIXME: hab noch nie verstanden warum das net einfach einfach ist...
        self.setLayout(self.widget_layout)

        self._op = None
        self._widgets = []
        self._par_names = []

    def _clear_widgets(self):
        """
        Remove any dynamically created widget
        """
        for i in reversed(range(self.widget_layout.count())):
            widget = self.widget_layout.takeAt(i).widget()
            if widget is not None:
                widget.deleteLater()
                widget.setParent(None)

        self._widgets = []
        self._par_names = []


    def _set_widgets(self):
        """
        Parse operation (function arguments) and create widgets according to
        the function annotations. Fill with default or current argument values
        and display the widgets.
        """
        self._clear_widgets()

        # Retreive operation (function detail), instantiate neccesary widgets
        package, fun = self._op["module"].rsplit('.', 1)
        module = import_module(package)
        self._op["function"] = getattr(module, fun)
        argspec = inspect.getfullargspec(self._op["function"])

        # Create widgets
        self._labels = []
        self._widgets = []
        self._none_checkbox_widgets = []
        for par_name, annotation in sorted(argspec.annotations.items()):
            label = QLabel(par_name)
            font = qtgui.QFont()
            font.setPointSize(10)
            label.setFont(font)

            # find associated widget and functions
            mapping = {}
            if "widget" in annotation:
                mapping["widget"] = widget = getattr(qtgui, annotation["widget"])
                mapping["display_func"] = annotation["display_func"]
                mapping["display_conversion"] = annotation["display_conversion"]
                mapping["get_func"] = annotation["get_func"]
            elif "type" in annotation:
                try:
                    mapping = dict(GenericOperationWidget.type_mapping[annotation["type"]])
                except KeyError:
                    print("Can't find wiget for type: {type} for parameter " +
                          "{par_name}".format(type=annotation["type"], par_name=par_name))
                    continue
            else:
                continue

            # instanciate widget and make display_function callable
            widget = mapping["widget"]()
            if not callable(mapping["display_func"]):
                mapping["display_func"] = widget.__getattribute__(mapping["display_func"])

            # save mapping and parameter name for later use
            widget.mapping = mapping
            widget.par_name = par_name

            # create checkbox to allow not to set this parameter
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            checkbox.setToolTip("Set the \"%s\" parameter. If unchecked, the default value is used." % par_name)
            checkbox.par_name = widget.par_name

            # Set value of widget according to parameters of the operation
            try:
                if "display_conversion" in mapping:
                    display_value = mapping["display_conversion"](self._op["kwargs"][par_name])    
                else:
                    display_value = self._op["kwargs"][par_name]
                mapping["display_func"](display_value)
            except KeyError:
                checkbox.setChecked(False)
            
            self._labels.append(label)
            self._widgets.append(widget)
            self._none_checkbox_widgets.append(checkbox)

        # Add widgets to layout
        if not self._widgets:
            self.widget_layout.addWidget(QLabel("No operation parameters"))
        for i, widget in enumerate(self._widgets):
            self.widget_layout.addWidget(self._labels[i])
            self.widget_layout.addWidget(self._widgets[i])
            self.widget_layout.addWidget(self._none_checkbox_widgets[i])

    def set_operation(self, op):
        """
        Create widgets for operation. Accepts either a full dict for a
        operation or a dict with only package, module and function
        """
        self._op = op
        self._set_widgets()

    def get_operation(self):
        """
        Returns operation dict for the current state of the parameter widgets.
        """
        kwargs = {}
        for i, w in enumerate(self._widgets):
            if hasattr(w, "par_name"):
                if self._none_checkbox_widgets[i].isChecked():
                    kwargs[w.par_name] = self.get_operation_parameter(w)

        self._op["kwargs"] = kwargs

        package, fun = self._op["module"].rsplit('.', 1)
        module = import_module(package)
        self._op["function"] = getattr(module, fun)
        return self._op
    
    def get_operation_parameter(self, w):       
        """
        Extract a parameter from a operations widget using mapping["get_func"]. 
        If a get_conversion is provided the raw input will be converted using 
        the get_conversion.Converts "" and "None" input to None for the time 
        being.
        
        Params:
        =======
        w : OperationsWidget
            The widget containing the parameter
            
        Returns:
        ========
        param : 
            Parameter as expected for the operation which w. represents
        """         
        try:
            # Try to get as many parameters as possible. If conversion fails it
            # will be apparent when the widget is repopulated as the parameter
            # is simply not set
            parameter = w.__getattribute__(w.mapping["get_func"])()
            if parameter == "" or parameter == "None":
                # FIXME: Move this to a generic get_conversion function
                parameter= None
            if "get_conversion" in w.mapping:
                parameter = w.mapping["get_conversion"](parameter)
        except ValueError:
            parameter = None
        return parameter
        


class Operation(QListWidgetItem):
    def __init__(self, operation, parent=None):
        QListWidgetItem.__init__(self, parent)
        self._op = operation
        self.setText(self._op["module"])

        
class TdmsChannelSelectWidget(QWidget):
    GROUP_PREFIX = "Read."

    def __init__(self, parent=None,
                 tdms_file=None,
                 group_label="Data group",
                 channel_labels=["X channel",
                                 "Y channel",
                                 "Z channel",
                                 "Z2 channel",
                                 ]):
        """
        Params
        ======
        twin_z : bool
            Allow to select two Z (data) channels
        group_label : str:
            Label text for group combo box
        channel_labels : list of str
            Label texts for channel combo boxes
        """
        QWidget.__init__(self, parent)
        self.widget_layout = QVBoxLayout()

        self.group_widgets = []
        self.channel_widgets = []
        for i, label_text in enumerate([group_label] + channel_labels):
            label = QLabel(label_text)
            channel_widget = QComboBox()
            channel_widget.addItem(label_text)
            channel_widget.setMinimumWidth(250)
            channel_widget.setDisabled(True)
            
            group_widget = QComboBox()
            group_widget.addItem(label_text)
            group_widget.setMinimumWidth(250)
            group_widget.setDisabled(True)

            layout = QHBoxLayout()
            layout.addWidget(label)
            layout.addWidget(group_widget)
            if i != 0:
                layout.addWidget(channel_widget)
            self.widget_layout.addLayout(layout)

            if i == 0:
                self.group_combo = group_widget
                self.group_combo.currentIndexChanged.connect(self.change_sub_channels)
            else:
                self.group_widgets.append(group_widget)
                self.channel_widgets.append(channel_widget)

                group_widget.currentIndexChanged.connect(self._populate_channels)


        self.setLayout(self.widget_layout)

        self.tdms_file = tdms_file

    @property
    def tdms_file(self):
        return self._tdms_file

    @tdms_file.setter
    def tdms_file(self, tdms_file):
        """
        Set the Tdms file in self.tdmsFiles at the specified index to be
        the currently used one and fill group and channel boxes appropriately)
        """
        self._tdms_file = tdms_file

        self._reset_channel_boxes()
        if tdms_file:
            self._populate_groups()
            self._populate_channels()

    def _reset_channel_boxes(self):
        for w in self.channel_widgets:
            w.clear()

    def _populate_groups(self, index=None):
        """
        Fill group_combo with the group names of the TDMS file that contain
        TdmsChannelSelectWidget.GROUP_PREFIX in their name. Try to reselect the
        item with the same index as selected before.

        Parameters
        ==========
        index: int
            unused, for compatibility with signals
        """
        self.blockSignals(True)
        old_index = self.group_combo.currentIndex()  

        for group_combo in ([self.group_combo] + self.group_widgets):
            group_combo.clear()
            for group in self.tdms_file.groups():
                if group.startswith(self.GROUP_PREFIX):
                    group_combo.addItem(group[len(self.GROUP_PREFIX):])
            group_combo.setEnabled(True)
            group_combo.setCurrentIndex(old_index)
            self.blockSignals(False)

    def _populate_channels(self, index=None):
        """
        Populate self.xChannelBox and self.yChannelBox with the channels
        of the selected group. If possible, use the channels selected before.

        Parameters
        ==========
        index: int
            unused, for compatibility with signals
        """
        for group_combo, channel_combo in zip(self.group_widgets, self.channel_widgets):
            
            group = self.GROUP_PREFIX + str(group_combo.currentText())
            channel_paths = [c.path for c in self.tdms_file.group_channels(group)]
    
            old_index = channel_combo.currentIndex()
        
            # Fill with new channels
            expr = re.compile(r"'/'(.+)'")
            channels = [expr.search(path).group(1) for path in channel_paths]
    
            channel_combo.clear()
            channel_combo.addItems(channels)
            channel_combo.setDisabled(False)
            channel_combo.setCurrentIndex(old_index)

    def get_group(self):
        """ Return selected group. No selection will return an empty string."""
        return self.GROUP_PREFIX + self.group_combo.currentText()

    def get_groups(self):
        """ Return selected group. No selection will return an empty string."""
        return [self.GROUP_PREFIX + w.currentText() for w in self.group_widgets]

    def get_channels(self):
        """
        Return list of selected channels (x, y, z and z2 if twin_x in was set
        to true). Channels where no selection has been made return an empy str.
        """
        return [w.currentText() for w in self.channel_widgets]
                
                
    def change_sub_channels(self):
        """
        Changes the channels according to the set top channel (group_combo)
        
        Parameters
        ==========
        index: int
            index to which the channels are set
        """
        new_index=self.group_combo.currentIndex()
        for group_combo in self.group_widgets:
            group_combo.setCurrentIndex(new_index)


class TdmsChannelSelectDialog(QDialog):
    def __init__(self, parent=None, **kwargs):
        QDialog.__init__(self)
        layout = QVBoxLayout()
        self.tdms_widget = TdmsChannelSelectWidget(parent=self, **kwargs)
        layout.addWidget(self.tdms_widget)

        # OK and Cancel buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)


    @staticmethod
    def get_group_channels(parent = None, **kwargs):
        """
        Params
        ======
        parent: None
        
        **kwargs : 
            Passed to TdmsChannelSelectWidget.__init__()

        Returns
        =======
        paths : list
            path (tuple of [group channel]) of all selected channels
        """
        dialog = TdmsChannelSelectDialog(parent, **kwargs)
        result = dialog.exec_()
        groups = dialog.tdms_widget.get_groups()
        channels = dialog.tdms_widget.get_channels()

        return list(zip(groups, channels)), result == QDialog.Accepted


if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = OperationsWidget()
    import bbFMR.processing as bp
    widget.populate_available_operations(list(filter(None, [("bbFMR.processing.%s" % s) if "__" not in s else None for s in dir(bp)])))
    ops = [{'module': "bbFMR.processing.limit"}]
    widget.set_operations(ops)
    widget.show()

#
#    from nptdms import TdmsFile
#    tdms_file = TdmsFile(r"H:\Auswertungen_Messungen\TAMR Bielefeld-Proben\Dm160727f\2017-01-12-DM160727f_MgO-E08-ip-impedance_jump-VNA-45Grad-0.1T_fw_highres.tdms")
#    TdmsChannelSelectDialog.get_group_channels(tdms_file=tdms_file,
#                                               group_label="Data group",
#                                               channel_labels=["Field",
#                                                               "Field_after",
#                                                               "Frequency",
#                                                               "Re(Signal)",
#                                                               "Im(Signal)"]
#                                               )
#
#
    sys.exit(app.exec_())
