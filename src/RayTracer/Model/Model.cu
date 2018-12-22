#include "Model.cuh"
CUDA_FUNC Triangle::Triangle(float3 p[3], float3 norm[3]) {
    pos[0] = p[0];
    pos[1] = p[1];
    pos[2] = p[2];
    normal[0] = norm[0];
    normal[1] = norm[1];
    normal[2] = norm[2];
}
CUDA_FUNC Triangle::Triangle(const float3 p[3], const float3 norm[3]) {
    pos[0] = p[0];
    pos[1] = p[1];
    pos[2] = p[2];
    normal[0] = norm[0];
    normal[1] = norm[1];
    normal[2] = norm[2];
}
CUDA_FUNC Triangle::Triangle(float3 a, float3 b, float3 c, float3 norm[3]) {
    pos[0] = a;
    pos[1] = b;
    pos[2] = c;
    normal[0] = norm[0];
    normal[1] = norm[1];
    normal[2] = norm[2];
}
CUDA_FUNC Triangle& Triangle::operator=(const Triangle& plus) {
    Triangle t1(plus.pos, plus.normal);
    return(t1);
}
CUDA_FUNC  bool  Triangle::hit(Ray r, IntersectRecord &colideRec) {

    colideRec.t = -1.0f;

    float3 ab, ac, ap, norm, e, qp;
    float t;
    ab = pos[1] - pos[0];
    ac = pos[2] - pos[0];
    qp = -r.getDir();
    norm = cross(ab, ac);
    float d = dot(qp, norm);
    if (d <= 0.0f) return false;
    ap = r.getOrigin() - pos[0];
    t = dot(ap, norm);
    if (t < 0.0f) return false;
    e = cross(qp, ap);
    float v = dot(ac, e);
    if (v < 0.0f || v > d) return false;
    float w = -dot(ab, e);
    if (w < 0.0f || v + w > d) return false;
    t /= d;

    colideRec.t = t;
    colideRec.normal = norm;
    colideRec.pos = r.getPos(t);
    return true;
}

__host__  bool Mesh::readFile(char * path) {
	ifstream file(path);

	vector<float3> vVertex;
	vector<float2> vText;
	vector<float3> vNorm;
	vector<vector<int3>> vFace;
	if (!file) {
		//	cerr << "Error::ObjLoader, could not open obj file:"
			//	<< path << " for reading." << endl;
		return false;
	}
	string line;
	while (getline(file, line)) {
		if (line.substr(0, 2) == "vt") {
			istringstream s(line.substr(2));
			float2 v;
			s >> v.x; s >> v.y;

			v.y = -v.y;
			vText.push_back(v);
		}
		else if (line.substr(0, 2) == "vn") {
			istringstream s(line.substr(2));
			float3 v;
			s >> v.x; s >> v.y; s >> v.z;

			vNorm.push_back(v);
		}
		else if (line.substr(0, 1) == "v") {
			istringstream s(line.substr(1));
			float3 v;
			s >> v.x; s >> v.y; s >> v.z;

			vVertex.push_back(v);
		}
		else if (line.substr(0, 1) == "f") {
			vector <int3> face;

			istringstream vtns(line.substr(1));
			string vtn;
			while (vtns >> vtn) {
				int3 vertex;
				replace(vtn.begin(), vtn.end(), '/', ' ');
				istringstream ivtn(vtn);
				if (vtn.find("  ") != string::npos) {
					ivtn >> vertex.x >> vertex.y;

					vertex.x--;
					vertex.y--;
					vertex.z = 0xfff;
				}
				else {
					ivtn >> vertex.x
						>> vertex.y
						>> vertex.z;


					vertex.x--;
					vertex.y--;
					vertex.z--;
				}
				face.push_back(vertex);
			}
			vFace.push_back(face);

		}
		else if (line[0] == '#') {
		}
		else {

		}
	}
	if (vFace.empty())
		return false;
	vector<Triangle> tempMesh;

	if (vText.size() != 0) {
		for (int f = 0; f < vFace.size(); f++) {
			int n = vFace[f].size();

			float3 V[3], N[3];
			for (int v = 0; v < n; v++) {
				int it = vFace[f][v].z;
				//	glTexCoord2f(vText[it].x, vText[it].y);

				int in = vFace[f][v].y;
				V[v].x = vNorm[in].x;
				V[v].y = vNorm[in].y;
				V[v].z = vNorm[in].z;

				int iv = vFace[f][v].x;
				N[v].x = vVertex[iv].x;
				N[v].y = vVertex[iv].y;
				N[v].z = vVertex[iv].z;
				//	glVertex3f(vVertex[iv].x, vVertex[iv].y, vVertex[iv].z);
			}
			Triangle t(V, N);
			tempMesh.push_back(t);
			//	glEnd();
		}
	}

	else {
		for (int f = 0; f < vFace.size(); f++) {
			int n = vFace[f].size();
			//	glBegin(GL_TRIANGLES);

			float3 V[3], N[3];
			for (int v = 0; v < n; v++) {
				int in = vFace[f][v].y;
				V[v].x = vNorm[in].x;
				V[v].y = vNorm[in].y;
				V[v].z = vNorm[in].z;

				int iv = vFace[f][v].x;
				N[v].x = vVertex[iv].x;
				N[v].y = vVertex[iv].y;
				N[v].z = vVertex[iv].z;
			}
			Triangle t(V, N);
			tempMesh.push_back(t);
			//	glEnd();
		}

	}
	cudaMalloc((void**)& meshTable, sizeof(Triangle)*(vFace.size() + 1));
	for (int i = 0; i < vFace.size(); i++) {
		meshTable[i] = tempMesh[i];
	}
	number = vFace.size();
	return true;
}

__host__ void mhr() {
	printf("dnmd");
}