import os


from . import railguards, retriever

llm_fn = railguards.llm_fns["open_ai_gpt_4o-mini"]

mau_don = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>Đơn khiếu nại</title>
    <script src="https://unpkg.com/pagedjs/dist/paged.polyfill.js"></script>
    <style>
        @page {
            size: A4 portrait;
            margin-top: 25mm;
            margin-bottom: 25mm;
            margin-left: 35mm;
            margin-right: 20mm;

            @top-center {
                content: counter(page);
                font-family: "Times New Roman", serif;
                font-size: 14pt;
                font-weight: normal;
                display: none;
            }

            @bottom-center {
                content: "";
            }
        }

        body {
            font-family: "Times New Roman", serif;
            font-size: 13pt;
            color: black;
            counter-reset: page;
        }

        /* Ẩn số trang trên trang đầu tiên */
        @page :first {
            @top-center {
                content: "";
            }
        }

        /* Hiển thị số trang từ trang thứ hai */
        @page :nth(2) {
            @bottom-center {
                content: counter(page);
                font-size: 14pt;
                text-align: center;
            }
        }

        /* Định dạng Quốc hiệu và Tiêu ngữ */
        .header {
            text-align: center;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .header p {
            margin: 0;
        }
        .header .line {
            width: 30%;
            border: 1px solid black;
            margin: 5px auto 0;
        }

        /* Định dạng Địa danh và ngày tháng */
        .location-date {
            text-align: right;
            margin: 5px 0 20px;
        }

        /* Định dạng Tiêu đề đơn */
        .document-title {
            text-align: center;
            font-weight: bold;
            text-transform: uppercase;
            font-size: 14pt;
            margin: 10px 0;
        }

        /* Định dạng Phần kính gửi */
        .recipient {
            margin-left: 1.5cm;
            margin-bottom: 20px;
        }
        .recipient p {
            margin: 0;
            font-weight: bold;
        }

        /* Định dạng Nội dung đơn */
        .content {
            text-align: justify;
            text-indent: 1.5cm;
            margin-bottom: 20px;
        }

        /* Định dạng Chữ ký người làm đơn */
        .signature {
            text-align: right;
            margin-top: 50px;
            margin-right: 1.5cm;
        }
        .signature p {
            margin: 0;
        }

    </style>
</head>
<body>
    <!-- Quốc hiệu và Tiêu ngữ -->
    <div class="header">
        <p>CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM</p>
        <p>Độc lập - Tự do - Hạnh phúc</p>
        <div class="line"></div>
    </div>

    <!-- Địa danh và ngày tháng -->
    <div class="location-date">
        Hà Nội, ngày ... tháng ... năm ...
    </div>

    <!-- Tiêu đề đơn -->
    <div class="document-title">
        ĐƠN KHIẾU NẠI
    </div>

    <!-- Kính gửi -->
    <div class="recipient">
        <p>Kính gửi:</p>
        <p>- [Tên cơ quan, tổ chức hoặc cá nhân có thẩm quyền giải quyết];</p>
    </div>

    <!-- Thông tin người khiếu nại -->
    <div class="content">
        <p>Tôi tên là: [Họ và tên người khiếu nại]</p>
        <p>Ngày sinh: [Ngày tháng năm sinh]</p>
        <p>CMND/CCCD số: [Số CMND/CCCD], cấp ngày [ngày cấp] tại [nơi cấp]</p>
        <p>Địa chỉ thường trú: [Địa chỉ]</p>
        <p>Số điện thoại liên hệ: [Số điện thoại]</p>
    </div>

    <!-- Nội dung khiếu nại -->
    <div class="content">
        <p>Nội dung khiếu nại:</p>
        <p>[Trình bày chi tiết sự việc, quyết định hành chính hoặc hành vi hành chính mà người khiếu nại cho rằng trái pháp luật, xâm phạm quyền và lợi ích hợp pháp của mình]</p>
    </div>

    <!-- Yêu cầu giải quyết -->
    <div class="content">
        <p>Yêu cầu giải quyết:</p>
        <p>[Nêu rõ yêu cầu của người khiếu nại đối với cơ quan, tổ chức hoặc cá nhân có thẩm quyền]</p>
    </div>

    <!-- Cam kết -->
    <div class="content">
        <p>Tôi cam kết những thông tin trên là đúng sự thật và chịu trách nhiệm trước pháp luật về những nội dung đã trình bày.</p>
    </div>

    <!-- Chữ ký người làm đơn -->
    <div class="signature">
        <p>Người làm đơn</p>
        <br><br><br>
        <p>(Ký, ghi rõ họ tên)</p>
    </div>
</body>
</html>"""

system_prompt = (
    "Bạn là một trợ lý ảo dành cho công dân viết đơn khiếu nại. Giúp tôi soạn đơn dựa vào các điều khoản có liên quan. "
    "Dùng mẫu đơn sau để soạn đơn: "
    f"{mau_don} "
    "nhớ trích dẫn tất cả các điều khoản quan trọng."
)
import json
import os
import subprocess
import tempfile


def recover_html(html):
    t1 = re.sub("(</div>)+$", "</body></html>", html)
    t1 = re.sub(
        '^\s*<div ".+?<meta charset',
        """<!DOCTYPE html>
    <html lang="vi">
    <head><meta charset""",
        t1,
    )
    if not ("<body>" in t1):
        t1 = re.sub("</style>", "</style></head><body>", t1)
    return t1


def save_and_generate_pdf(mau_don):
    mau_don_n1 = recover_html(mau_don)
    mau_don_new = mau_don_n1.replace(
        '<script src="https://unpkg.com/pagedjs/dist/paged.polyfill.js"></script>', ""
    )
    print(mau_don_n1)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_html:
        temp_html.write(mau_don_new.encode("utf-8"))
        temp_html_path = temp_html.name

    pdf_file_name = temp_html_path.replace(".html", ".pdf")
    try:
        subprocess.run(["pagedjs-cli", temp_html_path, "-o", pdf_file_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        os.remove(temp_html_path)

    return [mau_don, pdf_file_name]

import re


import gradio as gr
import mammoth

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

llm_fn_fixed = railguards.llm_fns["open_ai_gpt_4o-mini-fixed"]

def get_text(filename):
    doc = docx.Document(filename)
    return "\n".join([p.text for p in doc.paragraphs])


def parse_html_blob(response_text):
    """Extract code enclosed within triple backticks from the response."""
    pattern = r"```(?:html|htm)?\n(.*?)\n```"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return ""


def get_all_laws (fn):

    with open(fn) as f:
        return f.read()

all_laws = get_all_laws("/home/ubuntu/various_tools/troly_dontu/all_laws.txt")
law_index = retriever.index
async def get_related_regulations(qa):
    
    prompt = f"""trong các luật sau, luật nào có liên quan tới việc {qa}.
{all_laws}
    Trả lời dưới dạng một json list các luật có liên quan. 
    """
    
    applicable_laws = await railguards.get_chat_response(prompt, "", llm_fn_fixed)
    print(applicable_laws)
    applicable_laws = json.loads(re.sub(".*?```(json)?(.*?)```.*?$", r"\2", applicable_laws, flags=re.DOTALL))
    retrieved_nodes = []
    # resulted_laws = []
    for applicable_law in applicable_laws:
        try:
            if type(applicable_law) == dict and ("name" in applicable_law or "title" in applicable_law):
                if "name" in applicable_law:
                    applicable_law = applicable_law["name"]
                else:
                    applicable_law = applicable_law["title"]
                applicable_law = [x.strip('"') for x in all_laws.split("\n") if x.strip('"').startswith(applicable_law)][0]
            applicable_law = applicable_law.strip('"')
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="law",
                        operator=FilterOperator.EQ,
                        value=applicable_law.strip('"'),  # all_laws.split("\n")
                    )
                ]
            )
            my_retriever = VectorIndexRetriever(
                index=law_index,
                filters=filters,
                # You can adjust the number of results retrieved
                similarity_top_k=2,
            )
            #if os.path.exists(f"./txts/{applicable_law.replace('/','-')}.txt"):
            #    resulted_laws.append(f"./txts/{applicable_law.replace('/','-')}.txt")
        except:
            my_retriever = VectorIndexRetriever(
                index=law_index,
                #filters=filters,
                # You can adjust the number of results retrieved
                similarity_top_k=2,
            )
            print(applicable_law)
        nodes_re = await my_retriever.aretrieve(qa)
        retrieved_nodes.extend(nodes_re)
        
            #raise
    related_regulations = ""
    resulted_articles = []
    for node in retrieved_nodes:
        related_regulations += f"{node.text}\n({node.metadata['citation']})\n-------\n"
        related_file_path = f"./txts/{node.metadata['citation'].replace('/','')}.txt"
        if not os.path.exists(related_file_path):
            with open(related_file_path, "w") as f_related:
                f_related.write(node.text)
        resulted_articles.append(related_file_path)
    return related_regulations, resulted_articles


async def get_llm_answer(qa):    
    related_articles, list_of_regulations = await get_related_regulations(qa)
    my_system_prompt = (
        f"{system_prompt}\n Các điều sau có thể có liên quan: {related_articles}"
    )
    print (related_articles)
    
    #return await railguards.execute_chat_with_guardrail(
    #    qa, "đơn khiếu nại", system_prompt=my_system_prompt
    #), list_of_regulations
    return await railguards.get_chat_response(
        qa, system_prompt=my_system_prompt, llm_fn=llm_fn
    ), list_of_regulations


def get_references(history):
    return ["README.md"]


def make_content_editable(html):
    pattern = re.compile("<p>(.*?\[.+?\].*?</p>)")
    return pattern.sub(
        lambda match: f'<p contenteditable="true">{match.group(1)}', html
    )


async def get_html_from_llm(message, history):
    my_message = ""
    for historical_message in history:
        my_message += historical_message[0] +"\n"
    my_message += message
    print(my_message)
    response, list_of_files = await get_llm_answer(my_message)
    html = parse_html_blob(response)
    if html.strip() != "":
        response = response.replace(html, "xem bên cạnh")
    html = make_content_editable(html)

    return (response, list_of_files), html


async def add_message(history, message):
    history = history or []
    html_out = ""
    references = []
    if message:
        # history.append((message, None))
        if len(history) == 0:
            message = (
                "Làm đơn khiếu nại về việc " + message
                if "khiếu nại" not in message
                else message
            )
        (response, list_of_files), html_out = await get_html_from_llm(message, history)
        if html_out.strip() != "":
            response = response.replace(html_out, "xem bên cạnh")
            #_, pdf_filename = save_and_generate_pdf(html_out)
            references = list_of_files
        history.append((message, response))

    # references = get_references(history)

    return (
        history,
        gr.Textbox(value=None, interactive=False),
        html_out,
        references,
    )


def bot(history):
    
    return history


fake_run = True

js_code_update_html = """
(x) => {
    var e = document.getElementById("html_pdf");
    /* recover html */
    return [e.innerHTML,""];
}
"""
js_code_print_div = """
() =>{
        // Hide everything else
        document.querySelectorAll("body > *:not(.printable)").forEach(function (el) {
          el.classList.add("hidden");
        });

        // Use Paged.js for pagination
        
          window.print();

          // Restore hidden elements after printing
          document.querySelectorAll(".hidden").forEach(function (el) {
            el.classList.remove("hidden");
          });
        });
      }
"""
print_pdf_js_client_code="""
async () => {
            
            var printContent = document.getElementById('html_pdf');
            let html = '<html><body><p>Hello, world!</p></body></html>';
            
            //var WinPrint = window.open(url, '', 'width=900,height=650');
            let t1 = printContent.outerHTML;

            // 1. Replace one or more closing </div> tags at the end of the string
            //    with </body></html>
            t1 = t1.replace(/(<\/div>)+$/, "</body></html>");
        
            // 2. Replace the starting <div class=...> with the standard HTML header,
            //    unless it's followed by <meta>
            //console.log(t1);
            t1 = t1.replace(/^\s*<div .+<meta charset/s, "<!DOCTYPE html><html lang='vi'><head><meta charset");
            
            // 3. If the <body> tag is not present, insert it after </style>
            
            t1 = t1.replace(/<\/style>/, "</style></head><body>");
            console.log(t1);
            let blob = new Blob([t1], { type: 'text/html'});
            let url = URL.createObjectURL(blob);
            var WinPrint = window.open(url, '', 'width=900,height=650');
            
            //WinPrint.document.write(t1);
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            WinPrint.focus();
            WinPrint.print();
            WinPrint.close();
            
        }
"""
better_print_code = """
async () => {
    try {
        // Select the element containing the content to print
        const printContent = document.getElementById('html_pdf');
        if (!printContent) {
            throw new Error("Element with ID 'html_pdf' not found.");
        }

        // Extract the outer HTML of the selected element
        let modifiedHTML = printContent.outerHTML;

        // 1. Replace one or more closing </div> tags at the end with </body></html>
        modifiedHTML = modifiedHTML.replace(/(<\/div>)+$/, "</body></html>");

        // 2. Replace the starting <div class=...> with the standard HTML header,
        //    unless it's followed by <meta charset>
        modifiedHTML = modifiedHTML.replace(
            /^\s*<div .+<meta charset/s,
            "<!DOCTYPE html><html lang='vi'><head><meta charset");
        ;

        // 3. If the <body> tag is not present, insert it after </style>
        if (!/<\/?body>/i.test(modifiedHTML)) {
            modifiedHTML = modifiedHTML.replace(/<\/style>/i, "</style></head><body>");
        }

        // Create a Blob from the modified HTML
        const blob = new Blob([modifiedHTML], { type: 'text/html' });
        const blobURL = URL.createObjectURL(blob);

        // Open a new window with the Blob URL
        const printWindow = window.open(blobURL, '_blank', 'width=900,height=650');

        if (!printWindow) {
            throw new Error("Failed to open the print window. Please allow pop-ups for this website.");
        }

        // Define a handler to perform actions once the new window has loaded
        
            // Revoke the Blob URL to free up memory
        await new Promise(resolve => setTimeout(resolve, 2000));
        printWindow.focus();
        printWindow.print();
        printWindow.close();
        URL.revokeObjectURL(blobURL);
        //printWindow.focus();
        //printWindow.print();
        //printWindow.close();
        
        

    } catch (error) {
        // Handle any errors that occur during the process
        console.error("Error generating or printing PDF:", error);
        alert(`An error occurred: ${error.message}`);
    }
}
"""
head = """
<script src="https://unpkg.com/pagedjs/dist/paged.polyfill.js"></script>
"""
title="Trợ lý viết đơn khiếu nại"
description="""Ứng dụng viết đơn khiếu nại bằng GenAI. Ứng dụng sẽ tìm kiếm các điều khoản có liên quan tới sự việc, viết đơn, và hiển thị các điều khoản có liên quan.
            Hiện tại, mô hình được sử dụng là gpt-4o-mini (nhỏ, chất lượng chưa tối ưu, giá thành một câu trả lời chấp nhận được).
            Nhớ kiểm tra lại giải pháp của AI trước khi sử dụng.
            """
with gr.Blocks(fill_height=True) as demo:
    md = gr.Markdown(
    f"""
    # {title}
    {description}
    """)
    with gr.Row():
        # Left Column: Chatbot (approximately 1/4 of the page)
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                bubble_full_width=False,
                scale=1,
                placeholder="Trợ lý ảo đơn khiếu nại",
            )

            chat_input = gr.Textbox(
                interactive=True,
                placeholder="Nhập yêu cầu đơn",
                show_label=False,
            )
            submit_button = gr.Button("Gửi yêu cầu")

            gr.Examples(
                [
                    ["hàng xóm trổ cửa ra hẻm."],
                    ["ổ gà trên đường trước nhà không được khắc phục."],
                    ["hàng xóm xây tường lấn qua đất"],
                ],
                inputs=[chat_input],
                label="Làm đơn khiếu nại về việc ...",
            )

        with gr.Column(scale=2):
            gr.Markdown("## mẫu đơn")

            docx_display = gr.HTML(elem_id="html_pdf", elem_classes="printable")
            pdf_generation_button = gr.Button("In/Tạo file pdf")
        with gr.Column(scale=1):
            gr.Markdown("## Tài liệu tham khảo")
            tailieuthamkhao = gr.Files()
        chat_msg = submit_button.click(
            add_message,
            [chatbot, chat_input],
            [chatbot, chat_input, docx_display, tailieuthamkhao],
        )

        bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
        bot_msg.then(lambda: gr.Textbox(interactive=True), None, [chat_input])
        pdf_generation_button.click(
            None, None, None, js=better_print_code
        )
        '''
        pdf_generation_button.click(
            save_and_generate_pdf,
            [docx_display],
            [docx_display, tailieuthamkhao],
            js=js_code_update_html,
        )
        '''

demo.queue(5)