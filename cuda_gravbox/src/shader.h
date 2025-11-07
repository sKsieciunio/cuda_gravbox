#pragma once

#include <GL/glew.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class Shader
{
public:
	GLuint ID;

	//Shader(const char* vertexPath, const char* fragmentPath)
	//{
	//	std::string vertexCode;
	//	std::string fragmentCode;
	//	std::ifstream vShaderFile;
	//	std::ifstream fShaderFile;

	//	vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	//	fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

	//	try
	//	{
	//		vShaderFile.open(vertexPath);
	//		fShaderFile.open(fragmentPath);
	//		std::stringstream vShaderStream, fShaderStream;

	//		vShaderStream << vShaderFile.rdbuf();
	//		fShaderStream << fShaderFile.rdbuf();

	//		vShaderFile.close();
	//		fShaderFile.close();

	//		vertexCode = vShaderStream.str();
	//		fragmentCode = fShaderStream.str();
	//	}
	//	catch (std::ifstream::failure& e)
	//	{
	//		std::cout << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ: " << e.what() << std::endl;
	//	}

	//	const char* vShaderCode = vertexCode.c_str();
	//	const char* fShaderCode = fragmentCode.c_str();

	//	compileAndLink(vShaderCode, fShaderCode);
	//}

	static Shader FromSource(const char* vertexCode, const char* fragmentCode)
	{
		Shader shader;
		shader.compileAndLink(vertexCode, fragmentCode);
		return shader;
	}

	void use()
	{
		glUseProgram(ID);
	}

	void setMat4(const std::string& name, const float* value) const
	{
		glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, value);
	}

	void setFloat(const std::string& name, float value) const
	{
		glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
	}

private:
	Shader() : ID(0) {}

	void compileAndLink(const char* vShaderCode, const char* fShaderCode)
	{
		GLuint vertex, fragment;

		vertex = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertex, 1, &vShaderCode, NULL);
		glCompileShader(vertex);
		checkCompileErrors(vertex, "VERTEX");

		fragment = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragment, 1, &fShaderCode, NULL);
		glCompileShader(fragment);
		checkCompileErrors(fragment, "FRAGMENT");

		ID = glCreateProgram();
		glAttachShader(ID, vertex);
		glAttachShader(ID, fragment);
		glLinkProgram(ID);
		checkCompileErrors(ID, "PROGRAM");

		glDeleteShader(vertex);
		glDeleteShader(fragment);
	}

	void checkCompileErrors(GLuint shader, std::string type)
	{
		GLint success;
		GLchar infoLog[1024];
		if (type != "PROGRAM")
		{
			glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
			if (!success)
			{
				glGetShaderInfoLog(shader, 1024, NULL, infoLog);
				std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << std::endl;
			}
		}
		else
		{
			glGetProgramiv(shader, GL_LINK_STATUS, &success);
			if (!success)
			{
				glGetProgramInfoLog(shader, 1024, NULL, infoLog);
				std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << std::endl;
			}
		}
	}
};
