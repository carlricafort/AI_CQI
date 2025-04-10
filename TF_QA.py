# Install and Import the libraries/modules required if necessary
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from difflib import get_close_matches

# Predefined questions and contexts
QA_input = [
    {'question': 'What can you recommend if the Engineering Data Analysis (Lec) CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score Engineering Data Analysis (Lec) is less than 60%, give more activities and deeper lessons regarding Obtaining Data, Statistical Sampling, and Sampling Distributions.'},
    {'question': 'What to do if the Engineering Data Analysis (Lec) CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Engineering Data Analysis (Lec) is less than 60%, give more activities and deeper lessons regarding Point Estimation of Parameters, Probability Distributions, and Statistical Intervals.'},
    {'question': 'What can you suggest if the Engineering Data Analysis (Lec) CLO3 score is less than 60 %?',
    'context': 'If the CLO3 score in Engineering Data Analysis (Lec) is less than 60%, give more activities and deeper lessons regarding Hypothesis Testing, Regression and Correlation, and Design of Experiments.'},
    {'question': 'What can you recommend if the Calculus 1 CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Calculus 1 is less than 60%, give more activities and deeper lessons regarding Functions, Continuity, and Limits.'},
    {'question': 'What to do if the Calculus 1 CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Calculus 1 is less than 60%, give more activities and deeper lessons regarding Derivatives and Its Applications, and Higher-Order Derivatives.'},
    {'question': 'What can you suggest if the Calculus 1 CLO3 score is less than 60 %?',
    'context': 'If the CLO3 score in Calculus 1 is less than 60%, give more activities and deeper lessons regarding Parametric Equations and Partial Differentiation.'},
    {'question': 'What can you recommend if the Calculus 2 CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Calculus 2 is less than 60%, give more activities and deeper lessons regarding Integration Concept and Formulas, and Integration Techniques.'},
    {'question': 'What to do if the Calculus 2 CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Calculus 2 is less than 60%, give more activities and deeper lessons regarding Improper Integrals.'},
    {'question': 'What can you suggest if the Calculus 2 CLO3 score is less than 60 %?',
    'context': 'If the CLO3 score in Calculus 2 is less than 60%, give more activities and deeper lessons regarding Multiple Integration and Applications.'},
    {'question': 'What can you recommend if the Differential Equations CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Differential Equations is less than 60%, give more activities and deeper lessons regarding First Order, First Degree ODE, and Its Applications.'},
    {'question': 'What to do if the Differential Equations CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Differential Equations is less than 60%, give more activities and deeper lessons regarding Higher-Order ODE and Its Applications.'},
    {'question': 'What can you suggest if the Differential Equations CLO3 score is less than 60 %?',
    'context': 'If the CLO3 score in Differential Equations is less than 60%, give more activities and deeper lessons regarding Laplace Transforms, Inverses, and Its Applications.'},
    {'question': 'What can you recommend if the Advanced Engineering Mathematics (Lec) CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score Advanced Engineering Mathematics (Lec) is less than 60%, give more activities and deeper lessons regarding Simultaneous Linear and Nonlinear Equations, Complex Numbers and Its Applications.'},
    {'question': 'What to do if the Advanced Engineering Mathematics (Lec) CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Advanced Engineering Mathematics (Lec) is less than 60%, give more activities and deeper lessons regarding Power Series, Bessel, Legendre, Fourier Series, and Applications.'},
    {'question': 'What can you suggest if the Advanced Engineering Mathematics (Lec) CLO3 score is less than 60 %?',
    'context': 'If the CLO3 score in Advanced Engineering Mathematics (Lec) is less than 60%, give more activities and deeper lessons regarding Ordinary and Partial Differential Equations.'},
    {'question': 'What can you recommend if the Electromagnetics CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Electromagnetics is less than 60%, give more activities and deeper lessons regarding Vector Analysis.'},
    {'question': 'What to do if the Electromagnetics CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Electromagnetics is less than 60%, give more activities and deeper lessons regarding Directional Derivative, Gradient, Divergence, Curl, Integral Theorems.'},
    {"question": 'What can you suggest if the Electromagnetics CLO3 score is less than 60 %?',
    "context": "If the CLO3 score in Electromagnetics is less than 60%, give more activities and deeper lessons regarding Electric and Magnetic Fields, Dielectric and Magnetic Materials, Coupled and Magnetic Circuits, Time-varying Fields and Maxwell's Equation."},
    {'question': 'What can you recommend if the Electronics 1 (Lec) CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Electronics 1 (Lec) is less than 60%, give more activities and deeper lessons regarding Diode and Voltage Multipliers.'},
    {'question': 'What to do if the Electronics 1 (Lec) CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Electronics 1 (Lec) is less than 60%, give more activities and deeper lessons regarding BJT and FET.'},
    {"question": 'What can you suggest if the Electronics 1 (Lec) CLO3 score is less than 60 %?',
    "context": "If the CLO3 score in Electronics 1 (Lec) is less than 60%, give more activities and deeper lessons regarding BJT and FET Small Signal Analysis."},
    {'question': 'What can you recommend if the Electronics 2 (Lec) CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Electronics 2 (Lec) is less than 60%, give more activities and deeper lessons regarding BJT and FET Frequency Response.'},
    {'question': 'What to do if the Electronics 2 (Lec) CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Electronics 2 (Lec) is less than 60%, give more activities and deeper lessons regarding Cascade and Cascode Connections, Current Mirrors and Current Sources.'},
    {"question": 'What can you suggest if the Electronics 2 (Lec) CLO3 score is less than 60 %?',
    "context": "If the CLO3 score in Electronics 2 (Lec) is less than 60%, give more activities and deeper lessons regarding Differential Amplifiers, Operational Amplifiers, Feedback Systems, Oscillators, and Filters."},
    {'question': 'What can you recommend if the Electronics 3 (Lec) CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Electronics 3 (Lec) is less than 60%, give more activities and deeper lessons regarding SCR, UJT, PUT, TRIAC, DIAC, and other Thyristors, and Optoelectronic Devices.'},
    {'question': 'What to do if the Electronics 3 (Lec) CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Electronics 3 (Lec) is less than 60%, give more activities and deeper lessons regarding Sensors and Transducers, and Interfacing Techniques.'},
    {"question": 'What can you suggest if the Electronics 3 (Lec) CLO3 score is less than 60 %?',
    "context": "If the CLO3 score in Electronics 3 (Lec) is less than 60%, give more activities and deeper lessons regarding PLC and Building Management Systems."},
    {'question': 'What can you recommend if the Circuits 1 (Lec) CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Circuits 1 (Lec) is less than 60%, give more activities and deeper lessons regarding Resistive Network, Mesh and Node Equations.'},
    {'question': 'What to do if the Circuits 1 (Lec) CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Circuits 1 (Lec)  is less than 60%, give more activities and deeper lessons regarding Network Theorems.'},
    {"question": 'What can you suggest if the Circuits 1 (Lec) CLO3 score is less than 60 %?',
    "context": "If the CLO3 score in Circuits 1 (Lec) is less than 60%, give more activities and deeper lessons regarding Transient Analysis."},
    {'question': 'What can you recommend if the Circuits 2 (Lec) CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Circuits 2 (Lec) is less than 60%, give more activities and deeper lessons regarding Impedance, Admittance, and Resonance.'},
    {'question': 'What to do if the Circuits 2 (Lec) CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Circuits 2 (Lec) is less than 60%, give more activities and deeper lessons regarding Power in AC Circuits, Solutions to AC Network Problems.'},
    {"question": 'What can you suggest if the Circuits 2 (Lec) CLO3 score is less than 60 %?',
    "context": "If the CLO3 score in Circuits 2 (Lec) is less than 60%, give more activities and deeper lessons regarding Two-Port Network Parameters and Transfer Function."},
    {'question': 'What can you recommend if the Digital Electronics 1 (Lec) CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Digital Electronics 1 (Lec) is less than 60%, give more activities and deeper lessons regarding Boolean Algebra and Logic Gates.'},
    {'question': 'What to do if the Digital Electronics 1 (Lec) CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Digital Electronics 1 (Lec) is less than 60%, give more activities and deeper lessons regarding Minimization of Combinational Logic Circuits.'},
    {"question": 'What can you suggest if the Digital Electronics 1 (Lec) CLO3 score is less than 60 %?',
    "context": "If the CLO3 score in Digital Electronics 1 (Lec) is less than 60%, give more activities and deeper lessons regarding Algorithmic State Machine and Asynchronous Sequential Logic."},
    {'question': 'What can you recommend if the Digital Electronics 2 (Lec) CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Digital Electronics 2 (Lec) is less than 60%, give more activities and deeper lessons regarding Microprocessor Unit and Memory Subsystem.'},
    {'question': 'What to do if the Digital Electronics 2 (Lec) CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Digital Electronics 2 (Lec) is less than 60%, give more activities and deeper lessons regarding I/O Subsystem and Introduction to Set Architecture and Assembly Programming.'},
    {"question": 'What can you suggest if the Digital Electronics 2 (Lec) CLO3 score is less than 60 %?',
    "context": "If the CLO3 score in Digital Electronics 2 (Lec) is less than 60%, give more activities and deeper lessons regarding Microcontrollers."},
    {'question': 'What can you recommend if the Feedback and Control Systems (Lec) CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Feedback and Control Systems (Lec) is less than 60%, give more activities and deeper lessons regarding Pole-Zero Determination, System Modeling and Transfer Function.'},
    {'question': 'What to do if the Feedback and Control Systems (Lec) CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Feedback and Control Systems (Lec) is less than 60%, give more activities and deeper lessons regarding LTI Systems and Transient Response, Block Diagram, and Signal Flow Diagram.'},
    {"question": 'What can you suggest if the Feedback and Control Systems (Lec) CLO3 score is less than 60 %?',
    "context": "If the CLO3 score in Feedback and Control Systems (Lec) is less than 60%, give more activities and deeper lessons regarding Poles and Zeros, Root Locus, and Stability Analysis, Steady State Analysis and Frequency Response."},
    {'question': 'What can you recommend if the Communications 1 (Lec) CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Communications 1 (Lec) is less than 60%, give more activities and deeper lessons regarding Introduction to Communication Systems, Noise and dB Calculations.'},
    {'question': 'What to do if the Communications 1(Lec) CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Communications 1 (Lec) is less than 60%, give more activities and deeper lessons regarding AM, SSB Techniques, FM, Radio Receivers.'},
    {"question": 'What can you suggest if the Communications 1 (Lec) CLO3 score is less than 60 %?',
    "context": "If the CLO3 score in Communications 1 (Lec) is less than 60%, give more activities and deeper lessons regarding Radiation and Propagation Waves, Pulse Modulation, Digital Modulation, and Broadband Communication System."},
    {'question': 'What can you recommend if the Communications 2 (Lec) CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Communications 2 (Lec) is less than 60%, give more activities and deeper lessons regarding Basic Information Theory, Error Detection, Digital Communication.'},
    {'question': 'What to do if the Communications 2 (Lec) CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Communications 2 (Lec) is less than 60%, give more activities and deeper lessons regarding ASK, FSK, PSK, QAM.'},
    {"question": 'What can you suggest if the Communications 2 (Lec) CLO3 score is less than 60 %?',
    "context": "If the CLO3 score in Communications 2 (Lec) is less than 60%, give more activities and deeper lessons regarding Digital Transmission, Multiplexing, Frequency and Time Division Multiplexing."},
    {'question': 'What can you recommend if the Communications 3 (Lec) CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Communications 3 (Lec) is less than 60%, give more activities and deeper lessons regarding Basic Introduction to Data Communications.'},
    {'question': 'What to do if the Communications 3 (Lec) CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Communications 3 (Lec) is less than 60%, give more activities and deeper lessons regarding Category of Data Communications.'},
    {"question": 'What can you suggest if the Communications 3 (Lec) CLO3 score is less than 60 %?',
    "context": "If the CLO3 score in Communications 3 (Lec) is less than 60%, give more activities and deeper lessons regarding Configurations and Network Topology."},
    {'question': 'What can you recommend if the Communications 4 (Lec) CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Communications 4 (Lec) is less than 60%, give more activities and deeper lessons regarding Transmission Line, Matching Transmission Lines, and Smith Chart.'},
    {'question': 'What to do if the Communications 4 (Lec) CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Communications 4 (Lec) is less than 60%, give more activities and deeper lessons regarding Radio Wave Propagation, Power Density, and Field Strength Calculations.'},
    {"question": 'What can you suggest if the Communications 4 (Lec) CLO3 score is less than 60 %?',
    "context": "If the CLO3 score in Communications 4 (Lec) is less than 60%, give more activities and deeper lessons regarding Antenna Systems, Wave Guides, and Fiber Optics."},
    {'question': 'What can you recommend if the Signals, Spectra, and Signal Processing (Lec) CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Signals, Spectra, and Signal Processing (Lec) is less than 60%, give more activities and deeper lessons regarding Classification and Characteristics of Signals, Sampling Theorem and Aliasing.'},
    {'question': 'What to do if the Signals, Spectra, and Signal Processing (Lec) CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Signals, Spectra, and Signal Processing (Lec) is less than 60%, give more activities and deeper lessons regarding Convolution, Correlation, Fourier Series and Transform, Z-Transform.'},
    {"question": 'What can you suggest if the Signals, Spectra, and Signal Processing (Lec) CLO3 score is less than 60 %?',
    "context": "If the CLO3 score in Signals, Spectra, and Signal Processing (Lec) is less than 60%, give more activities and deeper lessons regarding Filtering and Difference Equations for FIR and IIR Filters."},
    {'question': 'What can you recommend if the Principles of ECE CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Principles of ECE is less than 60%, give more activities and deeper lessons regarding Algebra, Analytical and Solid Geometry, and Trigonometry.'},
    {'question': 'What to do if the Principles of ECE CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Principles of ECE is less than 60%, give more activities and deeper lessons regarding Basic Electronic Circuits.'},
    {"question": 'What can you suggest if the Principles of ECE CLO3 score is less than 60 %?',
    "context": "If the CLO3 score in Principles of ECE is less than 60%, give more activities and deeper lessons regarding Basic Circuit Analysis."},
    {'question': 'What can you recommend if the Principles of ICT CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Principles of ICT is less than 60%, give more activities and deeper lessons regarding Introduction to Web Development.'},
    {'question': 'What to do if the Principles of ICT CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Principles of ICT is less than 60%, give more activities and deeper lessons regarding Front-end Development.'},
    {"question": 'What can you suggest if the Principles of ICT CLO3 score is less than 60 %?',
    "context": "If the CLO3 score in Principles of ICT is less than 60%, give more activities and deeper lessons regarding Back-end Development."},
    {'question': 'What can you recommend if the Computer Aided Drafting CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Computer Aided Drafting is less than 60%, give more activities and deeper lessons regarding Introduction to CAD and Its Environment.'},
    {'question': 'What to do if the Computer Aided Drafting CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Computer Aided Drafting is less than 60%, give more activities and deeper lessons regarding Snapping and Construction Elements.'},
    {"question": 'What can you suggest if the Computer Aided Drafting CLO3 score is less than 60 %?',
    "context": "If the CLO3 score in Computer Aided Drafting is less than 60%, give more activities and deeper lessons regarding Dimensioning, Plotting, and Inputting."},
    {'question': 'What can you recommend if the Computer Programming 1 CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Computer Programming 1 is less than 60%, give more activities and deeper lessons regarding Introduction to Computers (Hardware and Software).'},
    {'question': 'What to do if the Computer Programming 1 CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Computer Programming 1 is less than 60%, give more activities and deeper lessons regarding History and Evolution of Computers, Algorithms.'},
    {"question": 'What can you suggest if the Computer Programming 1 CLO3 score is less than 60 %?',
    "context": "If the CLO3 score in Computer Programming 1 is less than 60%, give more activities and deeper lessons regarding Introduction to Computer Programming."},
    {'question': 'What can you recommend if the Computer Programming 2 CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Computer Programming 2 is less than 60%, give more activities and deeper lessons regarding Introduction to Embedded Systems, Hardware and Software Evolution.'},
    {'question': 'What to do if the Computer Programming 2 CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Computer Programming 2 is less than 60%, give more activities and deeper lessons regarding Arduino Environment.'},
    {"question": 'What can you suggest if the Computer Programming 2 CLO3 score is less than 60 %?',
    "context": "If the CLO3 score in Computer Programming 2 is less than 60%, give more activities and deeper lessons regarding Raspberry Pi Environment."},
    {'question': 'What can you recommend if the Environmental Science and Engineering CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Environmental Science and Engineering is less than 60%, give more activities and deeper lessons regarding Nature and Ecology, and Natural Systems and Resources.'},
    {'question': 'What to do if the Environmental Science and Engineering CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Environmental Science and Engineering is less than 60%, give more activities and deeper lessons regarding Environmental Concerns and Crises, Environmental Impact Assessment.'},
    {"question": 'What can you suggest if the Environmental Science and Engineering CLO3 score is less than 60 %?',
    "context": "If the CLO3 score in Environmental Science and Engineering is less than 60%, give more activities and deeper lessons regarding Sustainable Development."},
    {'question': 'What can you recommend if the Engineering Management CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Engineering Management is less than 60%, give more activities and deeper lessons regarding Evolution of Management Theory, Management and Its Function.'},
    {'question': 'What to do if the Engineering Management CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Engineering Management is less than 60%, give more activities and deeper lessons regarding Planning, Leading, Organizing, and Controlling.'},
    {"question": 'What can you suggest if the Engineering Management CLO3 score is less than 60 %?',
    "context": "If the CLO3 score in Engineering Management is less than 60%, give more activities and deeper lessons regarding Managing Product and Service Operations, and Managing the Marketing Function and Finance Function."},
    {'question': 'What can you recommend if theTechnopreneurship 101 CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Technopreneurship 101 is less than 60%, give more activities and deeper lessons regarding Technopreneurship Introduction, Customers, and Value.'},
    {'question': 'What to do if the Technopreneurship 101 CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Technopreneurship 101 is less than 60%, give more activities and deeper lessons regarding Proposition, Ethics, Social Responsibility and Globalization.'},
    {"question": 'What can you suggest if the Technopreneurship 101 CLO3  score is less than 60 %?',
    "context": "If the CLO3 score in Technopreneurship 101 is less than 60%, give more activities and deeper lessons regarding Business Models and Introduction to Intellectual Property, Execution and Business Plan, Financial Analysis and Accounting Basics, and Raising Capital."},
    {'question': 'What can you recommend if the Engineering Economy CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Engineering Economy is less than 60%, give more activities and deeper lessons regarding Introduction to Engineering Economics and Money-Time Relationship and Equivalence.'},
    {'question': 'What to do if the Engineering Economy CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Engineering Economy is less than 60%, give more activities and deeper lessons regarding Basic Economy Study Methods.'},
    {"question": 'What can you suggest if the Engineering Economy CLO3  score is less than 60 %?',
    "context": "If the CLO3 score in Engineering Economy is less than 60%, give more activities and deeper lessons regarding Decisions Under Certainty, Decisions Recognizing Risk, and Decisions Admitting Uncertainty."},
    {'question': 'What can you recommend if the ECE Laws, Contracts, Ethics, Standards & Safety CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in ECE Laws, Contracts, Ethics, Standards & Safety is less than 60%, give more activities and deeper lessons regarding Fundamentals of the Laws, Obligations, and Contracts, Regulation of ECE Profession, Practicing of the ECE Profession, and RA 9292.'},
    {'question': 'What to do if the ECE Laws, Contracts, Ethics, Standards & Safety CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in ECE Laws, Contracts, Ethics, Standards & Safety is less than 60%, give more activities and deeper lessons regarding Other ECE Relates Statutes such as NTC Memorandum Orders, IECEP, and PECs.'},
    {"question": 'What can you suggest if the ECE Laws, Contracts, Ethics, Standards & Safety CLO3  score is less than 60 %?',
    "context": "If the CLO3 score in ECE Laws, Contracts, Ethics, Standards & Safety is less than 60%, give more activities and deeper lessons regarding Safety Standards such as safety procedures in high risk activities and industries, and incident investigation and reporting."},
    {'question': 'What can you recommend if the Physics 2 CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Physics 2 is less than 60%, give more activities and deeper lessons regarding Thermodynamic, Atomic/Nuclear, and Condensed Matter2.'},
    {'question': 'What to do if the Physics 2 CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Physics 2 is less than 60%, give more activities and deeper lessons regarding Electricity, Magnetism, and Electromagnetic Induction, Inductance, and Alternating Circuits.'},
    {"question": 'What can you suggest if the Physics 2 CLO3  score is less than 60 %?',
    "context": "If the CLO3 score in Physics 2 is less than 60%, give more activities and deeper lessons regarding Optics."},
    {'question': 'What can you recommend if the Materials Science and Engineering CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Materials Science and Engineering is less than 60%, give more activities and deeper lessons regarding Modern Materials, Atomic Structure, and Interatomic Bonding, Crystalline and Non-crystalline Materials, Metals, and Alloys.'},
    {'question': 'What to do if the Materials Science and Engineering CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Materials Science and Engineering is less than 60%, give more activities and deeper lessons regarding Ceramics, Polymer Structures and Properties, Composites.'},
    {"question": 'What can you suggest if the Materials Science and Engineering CLO3  score is less than 60 %?',
    "context": "If the CLO3 score in Materials Science and Engineering is less than 60%, give more activities and deeper lessons regarding Electrical Properties, Dielectric Behavior, Magnetic Properties, Optical Properties, Environmental and Societal Issues in Materials Science and Engineering."},
    {'question': 'What can you recommend if the Total Quality Management CLO1 score is less than 60 %?',
    'context': 'If the CLO1 score in Total Quality Management is less than 60%, give more activities and deeper lessons regarding Quality, Leadership, Customer Satisfaction, and Employee Involvement.'},
    {'question': 'What to do if the Total Quality Management CLO2 score is less than 60 %?',
    'context': 'If the CLO2 score in Total Quality Management is less than 60%, give more activities and deeper lessons regarding Behavior and Communication in Teams, Quality Control Mnagement, Supplier Partnerships, Quality and Performance Measurement.'},
    {"question": 'What can you suggest if Total Quality Management CLO3  score is less than 60 %?',
    "context": "If the CLO3 score in Total Quality Management is less than 60%, give more activities and deeper lessons regarding Quality Tools and Techniques, Quality Management Sytems, Quality by Design, and Charting for Quality."}
]


# Load model and tokenizer
model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Continuous input loop
while True:
    user_question = input("\nType your question (or type 'x' to exit): ")
    if user_question.lower() == 'x':
        print("Exiting. Goodbye!")
        break


    # Find closest match from predefined questions
    question_texts = [item['question'] for item in QA_input]
    matches = get_close_matches(user_question, question_texts, n=1, cutoff=0.3)

    if matches:
        matched_question = matches[0]
        selected = next(item for item in QA_input if item['question'] == matched_question)

        # Run QA
        inputs = tokenizer(
            selected['question'],
            selected['context'],
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        outputs = model(**inputs)

        # Extract answer
        answer_start_idx = torch.argmax(outputs.start_logits)
        answer_end_idx = torch.argmax(outputs.end_logits)
        answer_tokens = inputs.input_ids[0, answer_start_idx: answer_end_idx + 1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

        # Show result
        print("\nMatched Question: {}".format(selected['question']))
        print("Answer: {}".format(answer))

    else:
        print("Sorry, I couldn't find a matching question.")
