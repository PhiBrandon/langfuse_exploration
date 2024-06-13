service_prompt = """
    Given a posting from a job listing website, identify the pain point of the job lister and generate the service offering from a professional that would alleviate that pain.
    The pain point and service offering should be concise, maximum of one sentence each.
    Generalize the service offering as much as possible.
    Output your answer in the structure and style of the examples. 


    Job Description:
    {job_description}

    Example Outputs:

    Service: Create a custom whiteboard animation explainer doodle video
    Pain Point: They want to leverage the engaging, explainer-style format of a whiteboard animation video to effectively communicate their ideas or showcase their products.

    Service: Find you a new job by searching and applying on your behalf
    Pain Point: Needs assistance with their job search, including identifying suitable opportunities and applying on their behalf, to secure a new role efficiently.

    Service: Do data analytics visualization power bi tableau dashboard
    Pain Point: Wants to leverage data analysis and visualization services to gain actionable insights from their raw data, leading to more informed decision-making for their business.

    Service: Do b2b lead generation database for targeted title person
    Pain Point: Needs a comprehensive, organized database of individuals categorized by their job titles or roles to streamline their data management and research efforts.

    Service: Cut cords of attachment with someone
    Pain Point: Seeks spiritual guidance and support to address emotional attachments and improve personal well-being.
    """

service_eval_prompt = """
    Given a posting from a job listing website and service offering, give a relevance score for the service offering.
    Job Description:
    {job_description}
    Service Offering:
    {offering}
    """

pain_eval_prompt = """
Given a posting from a job listing website and an identified pain point give a relevance score for the pain point.
    Job Description:
    {job_description}
    Pain Point:
    {pain}
    """
